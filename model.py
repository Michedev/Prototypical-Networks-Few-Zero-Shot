from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, MaxPool2d
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from path import Path

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

    def __init__(self, dataset: str, n: int, n_s: int, batch_size=32, lr=10e-3):
        super().__init__()
        self.lr = lr
        assert dataset in ['miniimagenet', 'omniglot']
        if dataset == 'miniimagenet':
            self.embedding_nn = embedding_omniglot()
        else:
            self.embedding_nn = embedding_miniimagenet()
        self.n = n
        self.n_s = n_s
        self.n_q = n - n_s
        self.loss_f = torch.nn.MSELoss(reduction='none')
        self.batch_size = batch_size


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
        embeddings_supp = embeddings_supp.view(batch_size, self.n_s, self.n, -1)
        c = embeddings_supp.mean(dim=1).detach()
        batch_query = batch_query.view(batch_query.size(0) *
                                       batch_query.size(1) *
                                       batch_query.size(2),
                                       batch_query.size(3),
                                       batch_query.size(4),
                                       batch_query.size(5))
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.view(batch_size, self.n_q, self.n, -1)
        return c, embeddings_query

    def training_step(self, batch):
        c, query = self(batch)
        loss = self.calc_loss(c, query)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def calc_loss(self, c, query):
        loss = 0.0
        for i in range(c.view(0)):
            for i_q in range(self.n_q):
                for i_cl in range(self.n):
                    loss += self.loss_f(query[i, i_q, i_cl], c[i, i_cl])
                    other_loss = 0.0
                    for j_cl in range(self.n):
                        if i_cl != j_cl:
                            other_loss += torch.exp(-self.loss_f(query[i, i_q, i_cl], c[i, j_cl]))
                    loss += other_loss.log()
        return loss

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            c, query = self(batch)
            loss = self.calc_loss(c, query)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

