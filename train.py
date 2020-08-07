from dataset import MiniImageNetDataLoader, pull_data_miniimagenet
from model import PrototypicalNetwork
from paths import ROOT, OMNIGLOTFOLDER, WEIGHTSFOLDER, EMBEDDING_PATH
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
import fire
import torch


def main(dataset, train_n: int, val_n: int, test_n: int, n_s: int, n_q: int, epochs: int = 1000, batch_size: int = 32,
         lr: float = 10e-3, trainsize: int = 10000, valsize: int = 64, testsize: int = 64,
         force_download=False, early_stop=False, gpu=0):
    assert dataset in ['omniglot', 'miniimagenet']
    EMBEDDING_PATH.replace('embedding', 'embedding_' + dataset)
    if dataset == 'omniglot':
        raise NotImplementedError("Omniglot not yet implemented")
    else:
        pull_data_miniimagenet(force_download)
        datamodule = MiniImageNetDataLoader(batch_size, train_n, val_n, test_n, n_s, n_q, trainsize, valsize, testsize)
    checkpoint = ModelCheckpoint(
        WEIGHTSFOLDER / 'best_model', verbose=True,
        mode='min', prefix=''
    )
    if early_stop:
        early_stop = pl.callbacks.EarlyStopping()
    trainer = pl.Trainer(checkpoint_callback=checkpoint, max_epochs=epochs,
                         early_stop_callback=early_stop, gpus=gpu,
                         auto_select_gpus=True)
    if not (WEIGHTSFOLDER / 'best_model.ckpt').exists():
        model = PrototypicalNetwork(train_n, test_n, n_s, n_q, lr)
    else:
        model = PrototypicalNetwork.load_from_checkpoint(WEIGHTSFOLDER / 'best_model.ckpt')
    trainer.fit(model, datamodule.train_dataloader(), datamodule.val_dataloader())
    trainer.test(model, datamodule.test_dataloader())


if __name__ == '__main__':
    fire.Fire(main)
