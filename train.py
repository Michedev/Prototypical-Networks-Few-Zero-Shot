from dataset import MiniImageNetMetaLearning, OmniglotMetaLearning, \
    get_train_test_classes, train_classes_miniimagenet, \
    val_classes_miniimagenet, test_classes_miniimagenet, \
    pull_data_miniimagenet, pull_data_omniglot
from model import PrototypicalNetwork
from paths import ROOT, OMNIGLOTFOLDER, WEIGHTSFOLDER, EMBEDDING_PATH
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar
import fire
import torch


def main(dataset, train_n, test_n, n_s, n_q, epochs=1000, batch_size=32, lr=10e-3, trainsize=10000, testsize=64,
         valsize=64, force_download=False, early_stop=False, gpu=0):
    assert dataset in ['omniglot', 'miniimagenet']
    EMBEDDING_PATH.replace('embedding', 'embedding_' + dataset)
    k = n_s + n_q
    if dataset == 'omniglot':
        raise NotImplementedError("Omniglot not yet implemented")
    else:
        pull_data_miniimagenet(force_download)
    checkpoint = ModelCheckpoint(
        WEIGHTSFOLDER / 'best_model.pth', verbose=True, mode='min',
    )
    trainer = pl.Trainer(checkpoint_callback=checkpoint, max_epochs=epochs,
                         early_stop_callback=early_stop, gpus=gpu, auto_select_gpus=True)
    model = PrototypicalNetwork(dataset, train_n, test_n, n_s, n_q, batch_size, lr, trainsize, valsize, testsize)
    if EMBEDDING_PATH.exists():
        model.embedding_nn.load_state_dict(torch.load(EMBEDDING_PATH))
    trainer.fit(model, )
    trainer.test()


if __name__ == '__main__':
    fire.Fire(main)
