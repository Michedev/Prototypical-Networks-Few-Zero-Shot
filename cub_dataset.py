import torch
from path import Path
from torch.utils.data import Dataset
from typing import Union, Literal, List, Tuple
from itertools import combinations


class CubDatasetEmbeddingsZeroShot(Dataset):
    """
    Cub Dataset which returns images refined from GoogleLeNet and class metadata.
    Image metadata have 1024 features while metadata have 312 features.
    """
    def __init__(self, root: Union[str, Path], split: Literal['train', 'val', 'test'],
                 query_size: int, num_classes: int):
        assert split in ['train', 'val', 'test']
        if isinstance(root, str):
            root = Path(root)
        root = root / 'cub'
        self.root = root
        self.num_classes = num_classes
        self.class_list = self._load_class_list(split)
        self.split = split
        self.label_attributes = self._load_global_class_attributes()
        self.folder_image_features: Path = self.root / 'images'
        self.query_size = query_size
        self.class_combinations = list(combinations(self.class_list, num_classes))
        self.query_index = [(i, j) for i in range(60) for j in range(10)]
        self.query_combinations = list(combinations(self.query_index, query_size))
        self.label_map = {c: i for i, c in enumerate(self.class_list)}
        self.label_global_index = {c: int(c[:3])-1 for c in self.class_list}

    def __len__(self):
        return len(self.class_combinations) * len(self.class_combinations)

    def __getitem__(self, i: int) -> dict:
        query_classes: Tuple[str] = self.class_combinations[i % len(self.class_combinations)]
        img_features = torch.zeros(self.query_size * self.num_classes, 1024)
        meta_features = torch.zeros(self.num_classes, 312)
        label = torch.zeros(self.query_size * self.num_classes)
        query_indexes: tuple = self.query_combinations[i % len(self.query_combinations)]
        counter = 0
        for i, class_fname in enumerate(query_classes):
            class_global_index = self.label_global_index[class_fname]
            meta_features[i] = self.label_attributes[class_global_index]
            class_image_features = torch.load(self.folder_image_features / class_fname)
            for i_row, i_split in query_indexes:
                label[counter] = i
                img_features[counter, :] = class_image_features[i_row, :, i_split]
                counter += 1
        return dict(test=(img_features, label), meta=meta_features)

    def _load_class_list(self, split: str) -> List[str]:
        fname = f'{split}classes.txt'
        path_file = self.root / fname
        with open(path_file, 'r') as f:
            classes = f.read()
        classes = [c for c in classes.split('\n') if c]
        return classes

    def _load_global_class_attributes(self):
        file_path = self.root / 'class_attribute_labels_continuos.txt'
        with open(file_path) as f:
            txt = f.read()
        attrs = [[float(num) for num in attr.split(' ')] for attr in txt.split('\n') if attr]
        attrs = torch.FloatTensor(attrs)
        assert attrs.shape == (200, 312), attrs.shape
        return attrs