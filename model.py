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

    def __init__(self, train_n: int, test_n: int, supp_size: int, query_size: int,
                 lr: float = 10e-3, distance_f: str = 'euclidean'):
        super().__init__()
        self.distance_f = distance_f
        self.lr = lr
        self.embedding_nn = embedding_miniimagenet()
        self.train_n = train_n
        self.test_n = test_n
        self.supp_size = supp_size
        self.query_size = query_size
        if distance_f == 'euclidean':
            self.distance_f = lambda x, y: (x - y).norm(2, dim=-1)

    def forward(self, batch_supp, y_supp, batch_query):
        batch_size, k_supp = batch_supp.shape[:2]
        k_query = batch_query.size(1)
        num_classes = y_supp[0].max() + 1
        batch_supp = batch_supp.reshape(batch_supp.size(0) *  # bs
                                        batch_supp.size(1),  # k
                                        batch_supp.size(2),  # channel
                                        batch_supp.size(3),  # w
                                        batch_supp.size(4))  # h
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.reshape(batch_size, num_classes, k_supp // num_classes, -1)
        c = embeddings_supp.mean(dim=2, keepdim=True).detach()
        batch_query = batch_query.reshape(batch_query.size(0) * # bs
                                          batch_query.size(1),  # k
                                          batch_query.size(2),  # channel
                                          batch_query.size(3),  # w
                                          batch_query.size(4))  # h
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.reshape(batch_size, num_classes, k_query // num_classes, -1)
        return c, embeddings_query

    def find_closest(self, c, query):
        return self.distance_f(query, c).unsqueeze(-1).argmax(-1)

    def training_step(self, batch, batch_bn):
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]

        c, query = self(train_inputs, train_targets, test_inputs)
        loss = self.calc_loss(c, query)
        with torch.no_grad():
            acc = self.calc_accuracy(c, query, test_targets)
        tensorboard_logs = {'loss/batch_train': loss, 'accuracy/batch_train': acc}
        pbar = {'acc': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs, 'progress_bar': pbar}

    def calc_accuracy(self, c, query, y_query):
        yhat = self.find_closest(c, query).flatten(1)
        acc = (yhat == y_query).float().mean()
        return acc

    def calc_loss(self, c: torch.Tensor, query: torch.Tensor):
        loss = torch.zeros([]).to(self.device)
        num_classes = query.size(1)
        sum_neg_distance = torch.zeros([]).to(self.device)
        for i_batch in range(c.shape[0]):
            for i_query in range(self.query_size):
                for i_class in range(num_classes):
                    for j_class in range(i_class, num_classes):
                        if i_class != j_class:
                            neg_distance = -self.distance_f(query[i_batch, i_class, i_query],
                                                            c[i_batch, j_class, 0])
                            neg_distance = neg_distance.exp() / (num_classes * self.query_size) + 10e-4
                            sum_neg_distance += neg_distance
                        else:
                            loss += self.distance_f(query[i_batch, i_class, i_query], c[i_batch, i_class, 0])
        loss += sum_neg_distance.log()
        return loss

    def training_epoch_end(self, outputs):
        torch.save(self.embedding_nn.state_dict(), EMBEDDING_PATH)
        avg_loss = sum(o['loss'] for o in outputs) / len(outputs)
        avg_acc = sum(o['acc'] for o in outputs) / len(outputs)
        print('\nTrain epoch loss:', avg_loss.item(),
              'Train epoch accuracy:', avg_acc.item())
        log = {'loss/epoch_train': avg_loss, 'accuracy/epoch_train': avg_acc}
        return {'train_loss': avg_loss, 'train_acc': avg_acc, 'log': log}

    def validation_step(self, batch, batch_nb):
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]

        c, query = self(train_inputs, train_targets, test_inputs)
        loss = self.calc_loss(c, query)
        acc = self.calc_accuracy(c, query, test_targets)
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
        train_inputs, train_targets = batch["train"]
        test_inputs, test_targets = batch["test"]

        c, query = self(train_inputs, train_targets, test_inputs)
        loss = self.calc_loss(c, query)
        acc = self.calc_accuracy(c, query, test_targets)
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
