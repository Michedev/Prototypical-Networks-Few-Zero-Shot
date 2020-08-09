from torch.utils.data import DataLoader
from torchmeta.datasets.helpers import miniimagenet, omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision import transforms

from paths import DATAFOLDER


def miniimagenet_transforms():
    return transforms.Compose([
        transforms.Resize(84),
        transforms.ToTensor(),
    ])


def get_meta_miniimagenet(n: int, supp_size: int, query_size: int, split: str):
    assert split in ['train', 'val', 'test']
    return miniimagenet(DATAFOLDER, supp_size, n, test_shots=query_size, meta_split=split)


def get_train_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'train')


def get_val_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'val')


def get_test_miniimagenet(n: int, supp_size: int, query_size: int):
    return get_meta_miniimagenet(n, supp_size, query_size, 'test')


def get_meta_omniglot(n: int, k: int, split: str):
    assert split in ['train', 'val', 'test']
    return omniglot(DATAFOLDER, k, n, meta_split=split)


def get_train_omniglot(n: int, k: int):
    return get_meta_omniglot(n, k, 'train')


def get_val_omniglot(n: int, k: int):
    return get_meta_omniglot(n, k, 'val')


def get_test_omniglot(n: int, k: int):
    return get_meta_omniglot(n, k, 'test')


class MiniImageNetDataLoader:

    def __init__(self, batch_size: int, train_n: int, val_n: int, test_n: int, n_s: int, n_q: int,
                 train_len: int, val_len: int, test_len: int, num_workers: int = None):
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.n_s = n_s
        self.n_q = n_q
        self.train_n = train_n
        self.val_n = val_n
        self.test_n = test_n
        self.batch_size = batch_size
        if num_workers is None:
            self.cpus = 0
        else:
            self.cpus = num_workers
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train_data = get_train_miniimagenet(self.train_n, self.n_s, self.n_q)
        self.val_data = get_val_miniimagenet(self.val_n, self.n_s, self.n_q)
        self.test_data = get_test_miniimagenet(self.test_n, self.n_s, self.n_q)

    def train_dataloader(self) -> DataLoader:
        return BatchMetaDataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.cpus)

    def val_dataloader(self):
        return BatchMetaDataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus)

    def test_dataloader(self) -> DataLoader:
        return BatchMetaDataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus)
