from datetime import datetime
from operator import itemgetter

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

from paths import LOGFOLDER, EMBEDDING_PATH
from tester import Tester


class Trainer(Tester):

    def __init__(self, model: Module, lr, epochs, device):
        loss_f = torch.nn.CrossEntropyLoss()
        super().__init__(model, loss_f, device)
        self.epochs = epochs
        self.lr = lr
        self.logger = SummaryWriter(LOGFOLDER / 'log_' + datetime.now().isoformat(sep='-'))

        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.opt, 3000, 0.5)

    def train_step(self, engine, batch):
        loss, acc = self.test_step(engine, batch)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        self.lr_scheduler.step()
        self.logger.add_scalar('loss/train_batch', loss, engine.state.iteration)
        self.logger.add_scalar('accuracy/train_batch', acc, engine.state.iteration)
        return loss, acc

    def train(self, train_loader, val_loader=None, train_len=None, val_len=None):
        trainer = Engine(self.train_step)

        RunningAverage(output_transform=itemgetter(0)).attach(trainer, 'train_loss')
        RunningAverage(output_transform=itemgetter(1)).attach(trainer, 'train_accuracy')

        ProgressBar().attach(trainer, ['train_loss', 'train_accuracy'])

        @trainer.on(Events.EPOCH_COMPLETED)
        def save_model(engine):
            torch.save(self.model.state_dict(), EMBEDDING_PATH)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_metrics(engine):
            train_loss_ = engine.state.metrics['train_loss']
            train_accuracy_ = engine.state.metrics['train_accuracy']
            print("Epoch", engine.state.epoch)
            print("Train loss", train_loss_, '-', 'Train Accuracy', train_accuracy_)
            self.logger.add_scalar('loss/epoch_train', train_loss_, engine.state.epoch)
            self.logger.add_scalar('accuracy/epoch_train', train_accuracy_, engine.state.epoch)

        if val_loader is not None:
            setup_validation(trainer, self.model, val_loader, self.logger, self.test_step, val_len)

        trainer.run(train_loader, self.epochs, train_len)


def setup_validation(trainer, model: Module, val_loader, logger, step_f, val_len):
    @trainer.on(Events.EPOCH_COMPLETED)
    def validate_data(engine: Engine):
        model.eval()

        val = Engine(step_f)

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

        with torch.no_grad():
            val.run(val_loader, 1, val_len)

        model.train()
