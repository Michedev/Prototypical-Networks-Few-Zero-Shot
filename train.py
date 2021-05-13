from argparse import ArgumentParser
from random import randint

import fire
import torch
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.transforms import Categorical, Rotation


def distance_f(name: str) -> torch.nn.Module:
    if name == 'euclidean':
        return torch.nn.MSELoss(reduction='none')
    elif name == 'cosine':
        return torch.nn.CosineEmbeddingLoss(reduction='none')
    else:
        raise ValueError(name)

def dataset_f(args):
    dataset = args.dataset
    dataset_kwargs = dict(folder=f'data/{dataset}',
                          shots=args.support_samples,
                          ways=args.classes,
                          shuffle=True,
                          test_shots=args.query_samples,
                          seed=args.dataset_seed,
                          meta_train=True,
                          target_transform=Categorical(num_classes=5),
                          )
    if dataset == 'omniglot':
       return omniglot(**dataset_kwargs, class_augmentations=[Rotation([90, 180, 270])],)
    elif dataset == 'miniimagenet':
        return miniimagenet(**dataset_kwargs)

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--dataset', '-d', required=True,
                           choices=['omniglot', 'miniimagenet'],
                           dest='dataset')
    argparser.add_argument('--classes', '-c', required=True, default=5, type=int, dest='classes')
    argparser.add_argument('--support-samples', '-s', required=True, default=1, type=int,
                           dest='support_samples')
    argparser.add_argument('--query-samples', '-q', required=True, default=5,
                           type=int, dest='query_samples')
    argparser.add_argument('--distance', '--dst', required=True,
                           default='euclidean', type=distance_f,
                           choices=['euclidean', 'cosine'])
    argparser.add_argument('--epochs', '-e', required=True, default=1_000)
    argparser.add_argument('--dataset-seed', default=None, type=int)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--batch-size', type=int, default=32)
    args = argparser.parse_args()
    if args.dataset_seed is None:
        args.dataset_seed = randint(0, 1_000_000)
        print('set dataset seed to %s' % args.dataset_seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('Cuda not available, fall back to cpu')
            args.device = 'cpu'
    return args


def main():
    args = parse_args()


if __name__ == '__main__':
    fire.Fire(main)
