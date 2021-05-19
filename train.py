from argparse import ArgumentParser, Namespace
from random import randint
from typing import Literal, Callable, Optional

from dataclasses import dataclass, field
import torch
import yaml
from path import Path
from torchmeta.datasets.helpers import omniglot, miniimagenet, cub
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, Rotation

from model import PrototypicalNetwork, PrototypicalNetworkZeroShot
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
                          meta_split=meta_split,
                          target_transform=Categorical(num_classes=args.num_classes),
                          download=True
                          )
    if dataset == 'omniglot':
        return omniglot(**dataset_kwargs, class_augmentations=[Rotation([90, 180, 270])], )
    elif dataset == 'miniimagenet':
        return miniimagenet(**dataset_kwargs)
    elif dataset.upper() == 'CUB':
        if args.support_samples == 0:
            from cub_dataset import CubDatasetEmbeddingsZeroShot
            return CubDatasetEmbeddingsZeroShot(DATAFOLDER, meta_split, args.query_samples, args.num_classes)
        else:
            return cub(**dataset_kwargs)

@dataclass
class Arguments(Namespace):
    dataset: str = field(init=False)
    num_classes: int = field(init=False)
    support_samples: int = field(init=False)
    query_samples: int = field(init=False)
    distance: Callable = field(init=False)
    epochs: int = field(init=False)
    seed: int = field(init=False)
    lr: float = field(init=False)
    weight_decay: float = field(init=False)
    use_lr_decay: bool = field(init=False)
    lr_decay_gamma: float = field(init=False)
    lr_decay_steps: int = field(init=False)
    device: str = field(init=False)
    batch_size: int = field(init=False)
    eval_steps: Optional[int] = field(init=False)
    run_path: Optional[Path] = field(init=False)
    epoch_steps: int = field(init=False)
    metadata_features: Optional[int] = field(init=False)
    image_features: Optional[int] = field(init=False)


def parse_args() -> Arguments:
    argparser = ArgumentParser()
    argparser.add_argument('--dataset', '-d',
                           choices=['omniglot', 'miniimagenet', 'cub'],
                           dest='dataset', help='Specify train dataset')
    argparser.add_argument('--classes', '--num-classes', '-c', default=5, type=int, dest='num_classes',
                           help='Number of classes for each task in meta learning i.e. the N in N-way with K shots')
    argparser.add_argument('--support-samples', '-s', default=1, type=int, dest='support_samples',
                           help='Number of training samples for each class in meta learning '
                                'i.e. the K in N-way with K shots')
    argparser.add_argument('--query-samples', '-q', default=5,
                           type=int, dest='query_samples',
                           help='Number of test samples for each class in meta learning')
    argparser.add_argument('--distance', '--dst',
                           default='euclidean', type=str,
                           choices=['euclidean', 'cosine'],
                           help='Distance function to use inside PrototypicalNetwork')
    argparser.add_argument('--epochs', '-e', default=500_000,
                           help='Number of training epochs. Set by default to a very high value '
                                'because paper specify that train continues until validation loss '
                                'continues to decrease.')
    argparser.add_argument('--epoch-steps', default=200, type=int, dest='epoch_steps')
    argparser.add_argument('--seed', default=13, type=int)
    argparser.add_argument('--device', type=str, default='cuda')
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--eval-steps', type=int, default=None,
                           help='Number of evaluation steps. '
                                'By default is set to the number '
                                'of steps to reach 600 episodes '
                                'considering batch size. This '
                                'is done to match paper results tables')
    argparser.add_argument('--run-path', type=Path, default=None,
                           help='Set this to resume a checkpoint '
                                'instead of start a new training.', dest='run_path')
    argparser.add_argument('--metadata-features', type=int, default=None,
                           help='Number of metadata features. Must set only for zero shot learning '
                                'i.e. when --support-samples=0', dest='metadata_features')
    argparser.add_argument('--image-features', type=int, default=None,
                           help='Number of image encoded features. Must set only for zero shot learning '
                                'i.e. when --support-samples=0', dest='image_features')
    argparser.add_argument('--lr', default=1e-4, type=float, help='lr for optimizer(adam)', dest='lr')
    argparser.add_argument('--weight-decay', default=0.0, type=float, dest='weight_decay')
    argparser.add_argument('--lr-decay', default=True, type=bool, dest='use_lr_decay',
                           help='Set true to use multiplicative lr decay '
                                '(set also --lr-decay-gamma and --lr-decay-steps)')
    argparser.add_argument('--lr-decay-gamma', default=None, type=float, dest='lr_decay_gamma',
                           help='Multiplicative factor to apply to lr decay')
    argparser.add_argument('--lr-decay-steps', default=None, type=int, dest='lr_decay_steps',
                           help='Number of steps to apply lr decay')
    args = argparser.parse_args(namespace=Arguments())

    if args.run_path is not None:
        run_path = Path(args.run_path)
        with open(run_path / 'config.yaml') as f:
            config = yaml.load(f)
        print('loaded config from', repr(run_path))
        args = Arguments()
        for k in config:
            setattr(args, k, config[k])  # args.k = config[k]

    if args.seed is None:
        args.seed = randint(0, 1_000_000)
        print('set seed to %s' % args.seed)

    if args.eval_steps is None:
        from math import ceil
        args.eval_steps = int(ceil(600 / args.batch_size))
        print('set eval_steps to %s' % args.eval_steps)

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
    dst_model_path: Path = experiment_path / 'model.py'
    src_model_path: Path = Path(__file__).abspath().parent / 'model.py'
    src_model_path.copy(dst_model_path)
    print('config:', yaml.safe_dump(args.__dict__), sep='\n')
    args.distance = distance_f(args.distance)
    args.device = torch.device(args.device)
    train_dataset = dataset_f(args, 'train')
    val_dataset = dataset_f(args, 'val')
    train_dloader = BatchMetaDataLoader(train_dataset, args.batch_size, shuffle=True)
    val_dloader = BatchMetaDataLoader(val_dataset, args.batch_size, shuffle=True)
    zero_shot = args.support_samples == 0
    if zero_shot:
        model = PrototypicalNetworkZeroShot(args.num_classes, False, args.metadata_features, args.image_features)
    else:
        input_channels = 1 if args.dataset == 'omniglot' else 3
        model = PrototypicalNetwork(args.num_classes, input_channels=input_channels)
    opt = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(model=model, train_dloader=train_dloader,
                      val_dloader=val_dloader,
                      distance_fun=args.distance,
                      run_path=experiment_path,
                      train_epochs=args.epochs, opt=opt,
                      use_lr_decay=args.use_lr_decay,
                      lr_decay_steps=args.lr_decay_steps,
                      lr_decay_gamma=args.lr_decay_gamma,
                      device=args.device,
                      eval_steps=args.eval_steps,
                      epoch_steps=args.epoch_steps,
                      zero_shot=zero_shot)
    trainer.train()


if __name__ == '__main__':
    main()
