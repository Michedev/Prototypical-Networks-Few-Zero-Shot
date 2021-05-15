from argparse import ArgumentParser, Namespace
from random import randint
from typing import Literal, Callable, Optional

from dataclasses import dataclass, field
import torch
import yaml
from path import Path
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, Rotation

from model import PrototypicalNetwork
from paths import new_experiment_path, DATAFOLDER
from trainer import Trainer
from utils import set_all_seeds


def distance_f(name: str) -> torch.nn.Module:
    if name == 'euclidean':
        return torch.nn.MSELoss(reduction='none')
    elif name == 'cosine':
        return torch.nn.CosineEmbeddingLoss(reduction='none')
    else:
        raise ValueError(name)


def dataset_f(args, meta_split: Literal['train', 'val', 'test'] = None):
    if meta_split is None:
        meta_split = 'train'
    dataset = args.dataset
    dataset_kwargs = dict(folder=DATAFOLDER,
                          shots=args.support_samples,
                          ways=args.num_classes,
                          shuffle=True,
                          test_shots=args.query_samples,
                          seed=args.seed,
                          meta_train=True,
                          target_transform=Categorical(num_classes=args.num_classes),
                          download=True
                          )
    if dataset == 'omniglot':
        return omniglot(**dataset_kwargs, class_augmentations=[Rotation([90, 180, 270])], )
    elif dataset == 'miniimagenet':
        return miniimagenet(**dataset_kwargs)


@dataclass
class Arguments(Namespace):
    dataset: str = field(init=False)
    num_classes: int = field(init=False)
    support_samples: int = field(init=False)
    query_samples: int = field(init=False)
    distance: Callable = field(init=False)
    epochs: int = field(init=False)
    seed: int = field(init=False)
    device: str = field(init=False)
    batch_size: int = field(init=False)
    eval_steps: Optional[int] = field(init=False)
    run_path: Optional[Path] = field(init=False)
    epoch_steps: int = field(init=False)


def parse_args() -> Arguments:
    argparser = ArgumentParser()
    argparser.add_argument('--dataset', '-d', required=True,
                           choices=['omniglot', 'miniimagenet'],
                           dest='dataset', help='Specify train dataset')
    argparser.add_argument('--classes', '--num-classes', '-c', required=True, default=5, type=int, dest='num_classes',
                           help='Number of classes for each task in meta learning i.e. the N in N-way with K shots')
    argparser.add_argument('--support-samples', '-s', required=True, default=1, type=int, dest='support_samples',
                           help='Number of training samples for each class in meta learning '
                                'i.e. the K in N-way with K shots')
    argparser.add_argument('--query-samples', '-q', default=5,
                           type=int, dest='query_samples',
                           help='Number of test samples for each class in meta learning')
    argparser.add_argument('--distance', '--dst',
                           default='euclidean', type=str,
                           choices=['euclidean', 'cosine'],
                           help='Distance function to use inside PrototypicalNetwork')
    argparser.add_argument('--epochs', '-e', default=1_000,
                           help='Number of training epochs. Set by default to a very high value '
                                'because paper specify that train continues until validation loss '
                                'continues to decrease.')
    argparser.add_argument('--epoch-steps', default=200, type=int, dest='epoch_steps')
    argparser.add_argument('--seed', default=None, type=int)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--eval-steps', type=int, default=None,
                           help='Number of evaluation steps. By default is set to the number of steps to reach 600 episodes considering batch size as paper reports')
    argparser.add_argument('--run-path', type=Path, default=None,
                           help='Set to resume a checkpoint', dest='run_path')
    args = argparser.parse_args(namespace=Arguments())
    if args.seed is None:
        args.seed = randint(0, 1_000_000)
        print('set seed to %s' % args.seed)
    if args.eval_steps is None:
        from math import ceil
        args.eval_steps = int(ceil(600 / args.batch_size))
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('Cuda not available, fall back to cpu')
            args.device = 'cpu'
    set_all_seeds(args.seed)
    return args


def main():
    args = parse_args()
    if args.run_path is None:
        experiment_path = new_experiment_path()
        print('Experiment run:', str(experiment_path))
    else:
        experiment_path = args.run_path
    with open(experiment_path / 'config.yaml', 'w') as f:
        yaml.dump(args.__dict__, f)
    print('config:', yaml.safe_dump(args.__dict__), sep='\n')
    args.distance = distance_f(args.distance)
    args.device = torch.device(args.device)
    train_dataset = dataset_f(args, 'train')
    val_dataset = dataset_f(args, 'val')
    train_dloader = BatchMetaDataLoader(train_dataset, args.batch_size, shuffle=True)
    val_dloader = BatchMetaDataLoader(val_dataset, args.batch_size, shuffle=True)
    input_channels = 1 if args.dataset == 'omniglot' else 3
    model = PrototypicalNetwork(args.num_classes, input_channels=input_channels)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    trainer = Trainer(model, train_dloader, val_dloader, args.distance,
                      experiment_path, args.epochs, opt, args.device,
                      args.eval_steps, args.epoch_steps)
    trainer.train()


if __name__ == '__main__':
    main()
