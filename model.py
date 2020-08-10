import pytorch_lightning as pl
import torch
from path import Path
from torch.nn import Conv2d, ReLU, Sequential, MaxPool2d, Flatten, GroupNorm

from paths import EMBEDDING_PATH


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


class ModelSaver:

    def __init__(self, model, savepath: Path, mode='min'):
        assert mode in ['min', 'max']
        assert savepath.endswith('.pth')
        self.model = model
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')
        self.savepath = savepath
        self.step = self.step_max if mode == 'max' else self.step_min

    def step_max(self, curr_value):
        if curr_value > self.best_value:
            self.best_value = curr_value
            torch.save(self.model.state_dict(), self.savepath)

    def step_min(self, curr_value):
        if curr_value < self.best_value:
            self.best_value = curr_value
            torch.save(self.model.state_dict(), self.savepath)

    def step(self, curr_value):
        raise NotImplementedError("Function initialized in the constructor")


class PrototypicalNetwork(pl.LightningModule):

    def __init__(self, train_n: int, test_n: int, supp_size: int, query_size: int, lr=10e-3, distance_f='euclidean'):
        super().__init__()
        self.lr = lr
        self.embedding_nn = embedding_miniimagenet()
        self.train_n = train_n
        self.test_n = test_n
        self.supp_size = supp_size
        self.query_size = query_size
        if distance_f == 'euclidean':
            self.distance_f = lambda x, y: (x - y).norm(2, dim=-1)
        self.loss_f = torch.nn.CrossEntropyLoss()

    def forward(self, batch_supp, y_train, batch_query):
        batch_size = batch_supp.size(0)
        num_classes = batch_supp.size(2)
        batch_supp = batch_supp.reshape(batch_supp.size(0) *  # bs
                                        batch_supp.size(1) *  # n_s
                                        batch_supp.size(2),  # n
                                        batch_supp.size(3),  # channel
                                        batch_supp.size(4),  # w
                                        batch_supp.size(5))  # h
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.reshape(batch_size, self.supp_size, num_classes, -1)
        c = torch.zeros(batch_size, num_classes, embeddings_supp.shape[-1]).to(self.device)
        for i_batch in range(batch_size):
            for i_supp in range(self.supp_size):
                for i_class in range(num_classes):
                    c[i_batch, y_train[i_batch, i_supp, i_class]] += embeddings_supp[i_batch, i_supp, i_class]
        c /= self.supp_size
        batch_query = batch_query.reshape(batch_query.size(0) *
                                          batch_query.size(1) *
                                          batch_query.size(2),
                                          batch_query.size(3),
                                          batch_query.size(4),
                                          batch_query.size(5))
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.reshape(batch_size, self.query_size, num_classes, -1)
        return c, embeddings_query

    def distances_centers(self, c, query):
        c = c.unsqueeze(1)
        query_reshaped = query.reshape(query.size(0), query.size(1) * query.size(2), 1, query.size(3))
        return self.distance_f(query_reshaped, c)
    
    def find_closest(self, c, query):
        return self.distances_centers(c, query).argmax(2)
    
    def training_step(self, batch, batch_bn):
        X_train, X_test, y_train, y_test = batch
        c, query = self(X_train, y_train, X_test)
        loss = self.calc_loss(c, query, y_test)
        with torch.no_grad():
            acc = self.calc_accuracy(c.detach(), query.detach(), y_test)
        tensorboard_logs = {'loss/batch_train': loss, 'accuracy/batch_train': acc}
        pbar = {'acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs, 'progress_bar': pbar}

    def calc_accuracy(self, c, query, y_test):
        pred_class = self.find_closest(c, query)
        y_test = y_test.flatten(1)
        return (pred_class == y_test).float().mean()

    def calc_loss(self, c: torch.Tensor, query: torch.Tensor, y_test):
        distances = self.distances_centers(c, query)
        distances = distances.softmax(dim=1)
        return self.loss_f(distances, y_test)

    def training_epoch_end(self, outputs):
        torch.save(self.embedding_nn.state_dict(), EMBEDDING_PATH)
        avg_loss = sum(o['loss'] for o in outputs) / len(outputs)
        avg_acc = sum(o['acc'] for o in outputs) / len(outputs)
        print('\nTrain epoch loss:', avg_loss.item(), 'Train epoch accuracy:', avg_acc.item())
        log = {'loss/epoch_train': avg_loss, 'accuracy/epoch_train': avg_acc}
        return {'train_loss': avg_loss, 'train_acc': avg_acc, 'log': log}

    def validation_step(self, batch, batch_nb):
        X_train, X_test, y_train, y_test = batch
        c, query = self(X_train, y_train, X_test)
        loss = self.calc_loss(c, query, y_test)
        acc = self.calc_accuracy(c, query, y_test)
        log = {'loss/val_epoch': loss, 'accuracy/val_epoch': acc}
        pbar = {'acc': acc}
        return {'val_loss': loss, 'val_acc': acc, 'log': log, 'progress_bar': pbar}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        print('Val epoch loss:', avg_loss.item(), 'Val epoch accuracy:', avg_acc.item())
        tensorboard_logs = {'loss/epoch_val': avg_loss, 'accuracy/epoch_val': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        X_train, X_test, y_train, y_test = batch
        c, query = self(X_train, y_train, X_test)
        loss = self.calc_loss(c, query, y_test)
        acc = self.calc_accuracy(c, query, y_test)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        print('Test epoch loss:', avg_loss, 'Test epoch accuracy:', avg_acc)
        tensorboard_logs = {'loss/test_epoch': avg_loss.item(), 'accuracy/test_epoch': avg_acc.item()}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        half_lr = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.5)
        return [opt], [half_lr]
