from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from path import Path
from dataset import MiniImageNetMetaLearning, train_classes_miniimagenet, \
    val_classes_miniimagenet, test_classes_miniimagenet
from multiprocessing import cpu_count
from paths import EMBEDDING_PATH


def EmbeddingBlock(input_channels):
    return Sequential(
        Conv2d(input_channels, 64, kernel_size=3),
        BatchNorm2d(64),
        ReLU(),
        MaxPool2d(2)
    )


def embedding_miniimagenet():
    return Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64)
    )


def embedding_omniglot():
    return EmbeddingBlock(3)


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

    def __init__(self, dataset: str, train_n: int, test_n: int, n_s: int, n_q: int, batch_size=32, lr=10e-3, train_length=None,
                 val_length=None,
                 test_length=None):
        super().__init__()
        self.lr = lr
        assert dataset in ['miniimagenet', 'omniglot']
        if dataset == 'miniimagenet':
            self.embedding_nn = embedding_omniglot()
        else:
            self.embedding_nn = embedding_miniimagenet()
        self.train_n = train_n
        self.test_n = test_n
        self.n_s = n_s
        self.n_q = n_q
        self.loss_f = torch.nn.MSELoss(reduction='none')
        self.batch_size = batch_size
        self.dataset = dataset
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        EMBEDDING_PATH.replace('embedding', 'embedding_'+dataset)

    def forward(self, X):
        batch_size = X.size(0)
        batch_supp = X[:, :self.n_s]
        batch_query = X[:, self.n_s:]
        batch_supp = batch_supp.view(batch_supp.size(0) *
                                     batch_supp.size(1) *
                                     batch_supp.size(2),
                                     batch_supp.size(3),
                                     batch_supp.size(4),
                                     batch_supp.size(5))
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.view(batch_size, self.n_s, self.train_n, -1)
        c = embeddings_supp.mean(dim=1).detach()
        batch_query = batch_query.view(batch_query.size(0) *
                                       batch_query.size(1) *
                                       batch_query.size(2),
                                       batch_query.size(3),
                                       batch_query.size(4),
                                       batch_query.size(5))
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.view(batch_size, self.n_q, self.train_n, -1)
        return c, embeddings_query

    def training_step(self, batch):
        c, query = self(batch)
        loss = self.calc_loss(c, query)
        tensorboard_logs = {'train_loss': loss}
        self.half_lr.step()
        return {'loss': loss, 'log': tensorboard_logs}

    def calc_accuracy(self, c, query):
        y_true = torch.arange(self.train_n).view(1, self.train_n).to(self.device)
        distancecs = (c.unsqueeze(1) - query).pow(2).sum(-1).sqrt()
        distancecs = distancecs.argmax(dim=1)
        acc = (y_true == distancecs).float().mean()
        return acc

    def calc_loss(self, c, query):
        loss = 0.0
        for i in range(c.view(0)):
            for i_q in range(self.n_q):
                for i_cl in range(self.train_n):
                    loss += self.loss_f(query[i, i_q, i_cl], c[i, i_cl])
                    other_loss = 0.0
                    for j_cl in range(self.train_n):
                        if i_cl != j_cl:
                            other_loss += torch.exp(-self.loss_f(query[i, i_q, i_cl], c[i, j_cl]))
                    loss += other_loss.log()
        return loss

    def training_epoch_end(self, outputs):
        torch.save(self.embedding_nn.state_dict(), EMBEDDING_PATH)

    def validation_step(self, batch, batch_nb):
        c, query = self(batch)
        loss = self.calc_loss(c, query)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        c, query = self(batch)
        loss = self.calc_loss(c, query)
        acc = self.calc_accuracy(c, query)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self) -> DataLoader:
        if self.dataset == 'miniimagenet':
            train_data = MiniImageNetMetaLearning(train_classes_miniimagenet(), self.train_n, self.n_s, self.n_q,
                                                  self.train_length)
        else:
            raise NotImplementedError("Omniglot data not implemented")
        return DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=cpu_count())

    def val_dataloader(self):
        if self.dataset == 'miniimagenet':
            val_data = MiniImageNetMetaLearning(val_classes_miniimagenet(), self.test_n, self.n_s, self.n_q, self.val_length)
        else:
            raise NotImplementedError("Omniglot data not implemented")
        return DataLoader(val_data, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def test_dataloader(self) -> DataLoader:
        if self.dataset == 'miniimagenet':
            test_data = MiniImageNetMetaLearning(test_classes_miniimagenet(), self.test_n, self.n_s, self.n_q,
                                                 self.test_length)
        else:
            raise NotImplementedError("Omniglot data not implemented")
        return DataLoader(test_data, batch_size=self.batch_size, shuffle=False, num_workers=cpu_count())

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.half_lr = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.5)
        return opt

