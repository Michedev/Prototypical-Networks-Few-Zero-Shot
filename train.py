import fire
import torch

from dataset import MiniImageNetDataLoader
from model import PrototypicalNetwork
from trainer import Trainer
from paths import EMBEDDING_PATH


def main(dataset, train_n: int, val_n: int, test_n: int, n_s: int, n_q: int, epochs: int = 1000, batch_size: int = 32,
         lr: float = 10e-3, trainsize: int = 10000, valsize: int = 64, testsize: int = 64, device='cpu'):
    """
    Train the model and save under model_weights both last epoch moodel (model_weights/embedding.pth) and
    the one with lowest validation loss (model_weights/best_embedding.pth)
    :param dataset: train dataset; can be 'omniglot' or 'miniimagenet' [str]
    :param train_n: num classes in train split (i.e. n in meta learning) [int]
    :param val_n: num classes in val split (i.e. n in meta learning) [int]
    :param test_n: num classes in test split (i.e. n in meta learning) [int]
    :param n_s: size of support set for each task (see paper for more details) [int]
    :param n_q: size of query set for each task (see paper for more details) [int]
    :param epochs: Num epochs of training [int]
    :param batch_size: Batch size [int]
    :param lr: learning rate [float]
    :param trainsize: Size of training set. Remember thought that instances are sampled randomly
    therefore it's useful only to set switch between training and validation. [int]
    :param valsize: Size of validation set. [int]
    :param testsize: Size of test set. [int]
    :param device: location of data and model parameters. Can be 'cpu' or 'cuda:*'
    """
    assert dataset in ['omniglot', 'miniimagenet']
    assert device == 'cpu' or 'cuda' in device
    print("Running in", device)
    if dataset == 'omniglot':
        raise NotImplementedError("Omniglot not yet implemented")
    else:
        datamodule = MiniImageNetDataLoader(batch_size, train_n, val_n, test_n, n_s, n_q, trainsize, valsize, testsize, device)
    model = PrototypicalNetwork().to(device)
    if EMBEDDING_PATH.exists():
        print('Loading', EMBEDDING_PATH)
        model.load_state_dict(torch.load(EMBEDDING_PATH, device))
    trainer = Trainer(model, lr, epochs, device)

    trainer.train(datamodule.train_dataloader(), datamodule.val_dataloader(), trainsize, valsize)


if __name__ == '__main__':
    fire.Fire(main)
