from dataclasses import dataclass

import numpy as np
import argparse
import json
from typing import List
import torch
import yaml
from path import Path
import re
from torchmeta.utils.data import BatchMetaDataLoader
from model import PrototypicalNetwork, PrototypicalNetworkZeroShot
from train import dataset_f, distance_f
from torch.utils.data import DataLoader
from trainer import Trainer


@dataclass
class TestParams:
    num_classes: int
    query_samples: int
    support_samples: int
    run_path: Path
    device: str
    batch_size: int
    dataset: str
    distance: str
    train_config: dict
    seed: int = 13
    steps: int = 600

def test_args() -> TestParams:
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--run-path', '-r', required=True, type=Path, dest='run_path')
    argparser.add_argument('--support-size', '-s', default=None, type=int, dest='support_samples')
    argparser.add_argument('--query-size', '-q', default=None, type=int, dest='query_samples')
    argparser.add_argument('--num-classes', '--nc', default=None, type=int, dest='num_classes')
    argparser.add_argument('--batch-size', '-b', default=32, type=int, dest='batch_size')
    argparser.add_argument('--seed', default=13, type=int, dest='seed')
    argparser.add_argument('--steps', default=600, type=int, dest='steps')
    argparser.add_argument('--device', '-d', default=None, type=str, dest='device')
    args = argparser.parse_args()

    with open(Path(args.run_path) / 'config.yaml') as f:
        train_config = yaml.safe_load(f)
    args.dataset = train_config['dataset']
    args.train_config = train_config
    args.distance = distance_f(args.train_config['distance'])
    return args


def load_best_model_checkpoint(run: Path, map_location=None):
    checkpoints: List[Path] = run.files('checkpoint_*')
    regex_score = re.compile(r"checkpoint\_(.+)\=([-,\d,\.]*)\.pt")
    score_checkpoints = [float(regex_score.fullmatch(c.basename()).group(2)) for c in checkpoints]
    i_best_checkpoint = np.argmax(score_checkpoints)
    best_checkpoint = checkpoints[i_best_checkpoint]
    model_parameters = torch.load(best_checkpoint, map_location=map_location)['model']
    return model_parameters


def main(args):
    test_dataset = dataset_f(args, 'test')
    if args.support_samples > 0:
        test_dloader = BatchMetaDataLoader(test_dataset, args.batch_size)
    else:
        test_dloader = DataLoader(test_dataset, args.batch_size)
    model_checkpoint = load_best_model_checkpoint(args.run_path, args.device)

    zero_shot = args.support_samples == 0
    if zero_shot:
        model = PrototypicalNetworkZeroShot(args.distance, num_classes=args.num_classes,
                                            meta_features=args.train_config['metadata_features'],
                                            img_features=args.train_config['image_features'])
    else:
        input_channels = 1 if args.dataset == 'omniglot' else 3
        model = PrototypicalNetwork(args.num_classes, input_channels=input_channels)
    model.load_state_dict(model_checkpoint)
    model = model.float().to(args.device)

    evaluator = Trainer(model, None, test_dloader, args.distance, args.run_path,
                        train_epochs=-1, opt=None, device=args.device,
                        eval_steps=args.steps, zero_shot=zero_shot)
    test_results = evaluator.eval()['val']
    print('evaluation done')
    eval_folder: Path = args.run_path / 'evaluation'
    eval_folder.mkdir_p()
    results = dict(num_classes=args.num_classes, support_samples=args.support_samples,
                   query_samples=args.query_samples, batch_size=args.batch_size,
                   steps=args.steps, eval_duration_seconds=test_results.eval_duration_seconds,
                   avg_accuracy=test_results.metrics['accuracy'], avg_loss=test_results.metrics['avg_loss'])
    dst_file = eval_folder / \
               f'num_classes={args.num_classes}_' \
               f'support={args.support_samples}_' \
               f'query={args.query_samples}.json'
    with open(dst_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('saved results to', dst_file)

if __name__ == '__main__':
    args = test_args()
    main(args)

