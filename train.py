import fire
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import MiniImageNetDataLoader, pull_data_miniimagenet
from model import PrototypicalNetwork, train_model
from paths import WEIGHTSFOLDER, EMBEDDING_PATH


def main(dataset, train_n: int, val_n: int, test_n: int, n_s: int, n_q: int, epochs: int = 1000, batch_size: int = 32,
         lr: float = 10e-3, trainsize: int = 10000, valsize: int = 64, testsize: int = 64,
         force_download=False, early_stop=False, device='cpu'):
    assert dataset in ['omniglot', 'miniimagenet']
    assert device == 'cpu' or 'cuda' in device
    EMBEDDING_PATH.replace('embedding', 'embedding_' + dataset)
    if dataset == 'omniglot':
        raise NotImplementedError("Omniglot not yet implemented")
    else:
        pull_data_miniimagenet(force_download)
        datamodule = MiniImageNetDataLoader(batch_size, train_n, val_n, test_n, n_s, n_q, trainsize, valsize, testsize, device)
    checkpoint = ModelCheckpoint(
        WEIGHTSFOLDER / 'best_model', verbose=True,
        mode='min', prefix=''
    )
    if early_stop:
        early_stop = pl.callbacks.EarlyStopping()
    trainer = pl.Trainer(checkpoint_callback=checkpoint, max_epochs=epochs,
                         early_stop_callback=early_stop, gpus=gpu,
                         auto_select_gpus=True)
    model = PrototypicalNetwork().to(device)
    if EMBEDDING_PATH.exists():
        print('Loading', EMBEDDING_PATH)
        model.load_state_dict(torch.load(EMBEDDING_PATH, device))

    train_model(model, lr, epochs, device, datamodule.train_dataloader(), datamodule.val_dataloader())


if __name__ == '__main__':
    fire.Fire(main)
