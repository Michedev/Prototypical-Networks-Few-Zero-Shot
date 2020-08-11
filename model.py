from datetime import datetime
from operator import itemgetter

import pytorch_lightning as pl
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from path import Path
from torch.nn import Conv2d, ReLU, Sequential, MaxPool2d, Flatten, GroupNorm, Module
from torch.utils.tensorboard import SummaryWriter
import ignite
from paths import EMBEDDING_PATH, LOGFOLDER


def EmbeddingBlock(input_channels):
    return Sequential(
        Conv2d(input_channels, 64, kernel_size=3, padding=1),
        GroupNorm(4, 64),
        ReLU(),
        MaxPool2d(2, ceil_mode=False)
    )


def embedding_miniimagenet():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        Flatten(start_dim=1)
    )


def embedding_omniglot():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        Flatten(start_dim=1)
    )


class PrototypicalNetwork(Module):

    def __init__(self, distance_f='euclidean'):
        super().__init__()
        self.embedding_nn = embedding_miniimagenet()
        if distance_f == 'euclidean':
            self.distance_f = lambda x, y: (x - y).norm(2, dim=-1)

    def forward(self, batch_supp, y_train, batch_query):
        batch_size, supp_size, num_classes = batch_supp.shape[:3]
        query_size = batch_query.size(1)
        batch_supp = batch_supp.reshape(batch_supp.size(0) *  # bs
                                        batch_supp.size(1) *  # n_s
                                        batch_supp.size(2),  # n
                                        batch_supp.size(3),  # channel
                                        batch_supp.size(4),  # w
                                        batch_supp.size(5))  # h
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.reshape(batch_size, supp_size, num_classes, -1)
        c = torch.zeros(batch_size, num_classes, embeddings_supp.shape[-1]).to(batch_supp.device)
        for i_batch in range(batch_size):
            for i_supp in range(supp_size):
                for i_class in range(num_classes):
                    c[i_batch, y_train[i_batch, i_supp, i_class]] += embeddings_supp[i_batch, i_supp, i_class]
        c /= supp_size
        batch_query = batch_query.reshape(batch_query.size(0) *
                                          batch_query.size(1) *
                                          batch_query.size(2),
                                          batch_query.size(3),
                                          batch_query.size(4),
                                          batch_query.size(5))
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.reshape(batch_size, query_size, num_classes, -1)
        return self.distances_centers(c, embeddings_query)

    def distances_centers(self, c, query):
        c = c.unsqueeze(1)
        query_reshaped = query.reshape(query.size(0), query.size(1) * query.size(2), 1, query.size(3))
        return self.distance_f(query_reshaped, c)


def train_model(model, lr, epochs, device, train_loader, val_loader=None):
    loss_f = torch.nn.CrossEntropyLoss()
    logger = SummaryWriter(LOGFOLDER / 'log_' + datetime.now().isoformat(sep='-'))

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 200, 0.5)

    def calc_loss(distances, y_test):
        distances = distances.reshape(distances.size(0) * distances.size(1), distances.size(2)).softmax(-1)
        y_test = y_test.flatten()
        return loss_f(distances, y_test)

    def calc_accuracy(distances, y_test):
        distances = distances.reshape(distances.size(0) * distances.size(1), distances.size(2)).argmax(-1)
        y_test = y_test.flatten(1)
        return (distances == y_test).float().mean()

    def train_step(engine, batch):
        loss, acc = test_step(batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
        lr_scheduler.step()
        logger.add_scalar('loss/train_batch', loss, engine.state.iteration)
        logger.add_scalar('accuracy/train_batch', acc, engine.state.iteration)
        return loss, acc

    def test_step(batch):
        X_train, X_test, y_train, y_test = batch
        X_train = X_train.to(device); X_test = X_test.to(device)
        y_train = y_train.to(device); y_test = y_test.to(device)
        distances = model(X_train, y_train, X_test)
        loss = calc_loss(distances, y_test)
        acc = calc_accuracy(distances, y_test)
        return loss, acc

    trainer = Engine(train_step)

    RunningAverage(output_transform=itemgetter(0)).attach(trainer, 'train_loss')
    RunningAverage(output_transform=itemgetter(1)).attach(trainer, 'train_accuracy')

    ProgressBar().attach(trainer, ['train_loss', 'train_accuracy'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_model(engine):
        torch.save(model.state_dict(), EMBEDDING_PATH)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(engine):
        train_loss_ = engine.state.metrics['train_loss']
        train_accuracy_ = engine.state.metrics['train_accuracy']
        print("Epoch", engine.state.epoch)
        print("Train loss", train_loss_, '-', 'Train Accuracy', train_accuracy_)
        logger.add_scalar('loss/epoch_train', train_loss_, engine.state.epoch)
        logger.add_scalar('accuracy/epoch_train', train_accuracy_, engine.state.epoch)

    if val_loader is not None:
        setup_validation(trainer, model, val_loader, logger, test_step)

    trainer.run(train_loader, epochs)


def setup_validation(trainer, model, val_loader, logger, step_f):
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_data(engine: Engine):
        model.eval()
        model.set_requires_grad_(False)

        val = Engine(lambda e, b: step_f(b))

        @val.on(Events.EPOCH_STARTED)
        def init_state(engine: Engine):
            engine.state.sum_loss = 0.0
            engine.state.sum_acc = 0.0

        @val.on(Events.ITERATION_COMPLETED)
        def sum_stats(engine):
            batch_loss, batch_acc = engine.state.output
            engine.state.sum_loss += batch_loss
            engine.state.sum_acc += batch_acc

        @val.on(Events.EPOCH_COMPLETED)
        def log_stats(engine):
            mean_loss = engine.state.sum_loss / engine.state.iteration
            mean_acc = engine.state.sum_acc / engine.state.iteration
            print("Validation Loss", float(mean_loss), '-', 'Validation Accuracy', float(mean_acc))
            logger.add_scalar('loss/epoch_val', mean_loss, trainer.state.epoch)
            logger.add_scalar('accuracy/epoch_val', mean_acc, trainer.state.epoch)

        val.run(val_loader, 1)

        model.train()
        model.set_requires_grad_(True)
