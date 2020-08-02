import zipfile
from multiprocessing import cpu_count
from typing import List
import pytorch_lightning as pl
import numpy as np
import torch
from random import sample, randint, shuffle

from path import Path
import wget
from skimage import io, transform
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_file_from_google_drive
from PIL import Image
from torchvision import transforms
import os
from paths import OMNIGLOTFOLDER, MINIIMAGENETFOLDER


class MetaLearningDataset(torch.utils.data.Dataset):

    def __init__(self, class_pool, n, n_s, n_q, random_rotation, image_size, length=None):
        self.class_pool = class_pool
        self.n = n
        self.k = n_s + n_q
        self.n_s = n_s
        self.n_q = n_q
        self.t = n * self.k
        self.ohe = torch.eye(n)
        self.image_size = list(image_size)
        self.random_rotation = random_rotation
        self.remaining_classes = []
        self.length = length
        self.preprocess_image = transforms.Compose([
            transforms.Resize(image_size[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.length is None:
            assert (len(self.class_pool) // self.n - 1) > 0

    def __len__(self):
        return len(self.class_pool) // self.n - 1 if self.length is None else self.length

    def __getitem__(self, i):
        if len(self.remaining_classes) < self.n:
            self.remaining_classes = self.class_pool[:]
        sampled_classes = sample(self.remaining_classes, self.n)
        for s_class in sampled_classes:
            self.remaining_classes.remove(s_class)
        n = len(sampled_classes)
        t = n * self.k
        X = torch.zeros([self.k, n] + self.image_size, dtype=torch.float32)
        image_names_batch, rotations, X = self.fit_train_task(X, sampled_classes, self.n_s)
        return X

    def shuffle(self):
        shuffle(self.class_pool)

    def fit_train_task(self, X, classes, n_s):
        image_names_batch = []
        rotations = {}
        for i_class, class_name in enumerate(classes):
            name_images = sample(class_name.files(), self.k)
            image_names_batch += name_images
            rotation = 0 if not self.random_rotation else 90 * randint(0, 3)
            rotations[i_class] = rotation
            for i_img, name_image in enumerate(name_images):
                img = self.load_and_transform(name_image, rotation)
                if X.shape[-1] == 1:
                    img = img.unsqueeze(-1)
                X[i_img, i_class, :, :, :] = img
                del img
        return image_names_batch, rotations, X

    def load_and_transform(self, name_image, rotation):
        img = Image.open(name_image).convert('RGB')
        if rotation != 0:
            img = img.rotate(rotation)
        return self.preprocess_image(img)


class OmniglotMetaLearning(MetaLearningDataset):

    def __init__(self, class_pool, n, n_s, n_q, length=None):
        super(OmniglotMetaLearning, self).__init__(class_pool, n, n_s, n_q, random_rotation=True,
                                                   image_size=[1, 28, 28], length=length)


class MiniImageNetMetaLearning(MetaLearningDataset):

    def __init__(self, class_pool, n, n_s, n_q, length=None):
        super(MiniImageNetMetaLearning, self).__init__(class_pool, n, n_s, n_q, False, image_size=[3, 84, 84],
                                                       length=length)


def get_train_test_classes(classes: list, test_classes_file: Path, train_classes_file: Path, numtrainclasses):
    if not train_classes_file.exists() or not test_classes_file.exists():
        index_classes = np.arange(len(classes))
        np.random.shuffle(index_classes)
        index_train = index_classes[:numtrainclasses]
        index_test = index_classes[numtrainclasses:]
        train_classes = [classes[i_train] for i_train in index_train]
        test_classes = [classes[i_test] for i_test in index_test]
    else:
        with open(train_classes_file) as f:
            train_classes = f.read()
        train_classes = train_classes.split(', ')
        train_classes = list(map(Path, train_classes))

        with open(test_classes_file) as f:
            test_classes = f.read()
        test_classes = test_classes.split(', ')
        test_classes = list(map(Path, test_classes))
    return train_classes, test_classes


def pull_data_omniglot(force):
    if force or not OMNIGLOTFOLDER.exists():
        archives = ['images_background', 'images_evaluation']
        for archive_name in archives:
            wget.download(f'https://github.com/brendenlake/omniglot/raw/master/python/{archive_name}.zip')
        if OMNIGLOTFOLDER.exists():
            for el in OMNIGLOTFOLDER.files(): el.remove()
        OMNIGLOTFOLDER.makedirs_p()
        for archive in archives:
            with zipfile.ZipFile(f'{archive}.zip') as z:
                z.extractall(OMNIGLOTFOLDER)
            Path(f'{archive}.zip').remove()
        for idiom_folder in OMNIGLOTFOLDER.dirs():
            for char_folder in idiom_folder.dirs():
                char_folder.move(OMNIGLOTFOLDER)
        for folder in OMNIGLOTFOLDER.dirs('images_*'):
            folder.removedirs()


def pull_data_miniimagenet(force):
    test_id = '1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v'
    val_id = '1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl'
    train_id = '107FTosYIeBn5QbynR46YG91nHcJ70whs'
    if not MINIIMAGENETFOLDER.exists():
        MINIIMAGENETFOLDER.makedirs()
    for zipfname, url in [('train.tar', train_id), ('val.tar', val_id), ('test.tar', test_id)]:
        tarfile = MINIIMAGENETFOLDER / zipfname
        dstfolder = MINIIMAGENETFOLDER / zipfname.split('.')[0]
        if not dstfolder.exists() or force:
            download_file_from_google_drive(url, MINIIMAGENETFOLDER, zipfname)
            if dstfolder.exists() and force:
                dstfolder.removedirs()
            os.system(f'tar -xvf {tarfile} --directory {MINIIMAGENETFOLDER}')
            tarfile.remove()


def train_classes_miniimagenet():
    return (MINIIMAGENETFOLDER / 'train').dirs()


def val_classes_miniimagenet():
    return (MINIIMAGENETFOLDER / 'val').dirs()


def test_classes_miniimagenet():
    return (MINIIMAGENETFOLDER / 'test').dirs()


class MiniImageNetDataLoader:

    def __init__(self, batch_size: int, train_n: int, val_n: int, test_n: int, n_s: int, n_q: int,
                 train_len: int, val_len: int, test_len: int, cpus: int = None):
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len
        self.n_s = n_s
        self.n_q = n_q
        self.train_n = train_n
        self.val_n = val_n
        self.test_n = test_n
        self.batch_size = batch_size
        if cpus is None:
            self.cpus = cpu_count()
        else:
            self.cpus = cpus
        self.prepare_data()

    def prepare_data(self) -> None:
        self.train_data = MiniImageNetMetaLearning(train_classes_miniimagenet(), self.train_n,
                                                   self.n_s, self.n_q, self.train_len)
        self.val_data = MiniImageNetMetaLearning(val_classes_miniimagenet(), self.val_n,
                                                 self.n_s, self.n_q, self.val_len)
        self.test_data = MiniImageNetMetaLearning(test_classes_miniimagenet(), self.test_n,
                                                  self.n_s, self.n_q, self.test_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.cpus)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.cpus)
