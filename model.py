import torch
from torch import nn

from paths import EMBEDDING_PATH


def EmbeddingBlock(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, ceil_mode=True)
    )


def embedding_module():
    return nn.Sequential(
        EmbeddingBlock(3),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        EmbeddingBlock(64),
        nn.Flatten(start_dim=1)
    )


class PrototypicalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding_nn = embedding_module()

    def forward(self, X):
        batch_size = X.size(0)
        batch_supp = X[:, :self.supp_size]
        batch_query = X[:, self.supp_size:]
        batch_supp = batch_supp.flatten(0, 1)
        embeddings_supp = self.embedding_nn(batch_supp)
        embeddings_supp = embeddings_supp.reshape(batch_size, self.supp_size, self.train_n, -1)
        centroids = embeddings_supp.mean(dim=1).detach()
        batch_query = batch_query.flatten(0, 1)
        embeddings_query = self.embedding_nn(batch_query)
        embeddings_query = embeddings_query.reshape(batch_size, self.query_size, self.train_n, -1)
        return centroids, embeddings_query

    def training_step(self, batch, batch_bn):
        X, y = batch
        c, query = self(X)
        loss = self.calc_loss(c, query)
        with torch.no_grad():
            acc = self.calc_accuracy(c.detach(), query.detach())
        tensorboard_logs = {'loss/batch_train': loss, 'accuracy/batch_train': acc}
        return {'loss': loss, 'acc': acc, 'log': tensorboard_logs}

    def calc_accuracy(self, c, query):
        y_true = torch.arange(self.train_n).reshape(1, self.train_n).to(self.device)
        distancecs = (c.unsqueeze(1) - query).pow(2).sum(-1).sqrt()
        distancecs = distancecs.argmax(dim=1)
        acc = (y_true == distancecs).float().mean()
        return acc

    def calc_loss(self, c, query):
        loss = 0.0
        for i in range(c.size(0)):
            for i_q in range(self.query_size):
                for i_cl in range(self.train_n):
                    loss += self.loss_f(query[i, i_q, i_cl], c[i, i_cl]).sum()
                    other_loss = 0.0
                    for j_cl in range(self.train_n):
                        if i_cl != j_cl:
                            other_loss += torch.exp(-self.loss_f(query[i, i_q, i_cl], c[i, j_cl])).sum()
                    loss += other_loss.log()
        return loss

    def training_epoch_end(self, outputs):
        torch.save(self.embedding_nn.state_dict(), EMBEDDING_PATH)
        avg_loss = sum(o['loss'] for o in outputs) / len(outputs)
        avg_acc = sum(o['acc'] for o in outputs) / len(outputs)
        log = {'loss/epoch_train': avg_loss, 'accuracy/epoch_train': avg_acc}
        return {'train_loss': avg_loss, 'train_acc': avg_acc, 'log': log}

    def validation_step(self, batch, batch_nb):
        X, y = batch
        c, query = self(X)
        loss = self.calc_loss(c, query)
        acc = self.calc_accuracy(c, query)
        log = {'loss/val_epoch': loss, 'accuracy/val_epoch': acc}
        return {'val_loss': loss, 'val_acc': acc, 'log': log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'loss/epoch_val': avg_loss, 'accuracy/epoch_val': avg_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        X, y = batch
        c, query = self(X)
        loss = self.calc_loss(c, query)
        acc = self.calc_accuracy(c, query)
        return {'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        tensorboard_logs = {'loss/test_epoch': avg_loss, 'accuracy/test_epoch': avg_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        half_lr = torch.optim.lr_scheduler.StepLR(opt, 2000, 0.5)
        return [opt], [half_lr]
