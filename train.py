from dataset import MiniImageNetMetaLearning, OmniglotMetaLearning, \
                    get_train_test_classes, train_folders_miniimagenet, \
                    val_folders_miniimagenet, test_folders_miniimagenet, \
                    pull_data_miniimagenet, pull_data_omniglot
from _train import train
from paths import ROOT, OMNIGLOTFOLDER


def main(dataset, n, n_s, n_q, epochs=1000, force_download=False):
    assert dataset in ['omniglot', 'miniimagenet']
    k = n_s + n_q
    if dataset == 'omniglot':
        pull_data_omniglot(force_download)
        classes = list(OMNIGLOTFOLDER.glob('*/*/'))
        train_classes_file = ROOT / f'train_classes_{dataset}.txt'
        test_classes_file = ROOT / f'test_classes_{dataset}.txt'
        train_classes, test_classes = \
            get_train_test_classes(classes, test_classes_file, train_classes_file, 1200)
    else:
        pull_data_miniimagenet(force_download)
        train_classes = train_folders_miniimagenet()
        val_classes = val_folders_miniimagenet()
        test_classes = test_folders_miniimagenet()
        train_data = MiniImageNetMetaLearning(train_data)
    
