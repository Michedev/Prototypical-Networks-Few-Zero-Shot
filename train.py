from argparse import ArgumentParser
from dataclasses import dataclass
from random import randint
from typing import Optional

import fire
import torch
import yaml
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Average, RunningAverage, Accuracy
from path import Path
from torch.utils.data import DataLoader
from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.transforms import Categorical, Rotation

from paths import new_experiment_path


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
        return omniglot(**dataset_kwargs, class_augmentations=[Rotation([90, 180, 270])], )
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
    argparser.add_argument('--val-steps', type=int, default=None)
    args = argparser.parse_args()
    if args.dataset_seed is None:
        args.dataset_seed = randint(0, 1_000_000)
        print('set dataset seed to %s' % args.dataset_seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print('Cuda not available, fall back to cpu')
            args.device = 'cpu'
    return args


@dataclass
class Trainer:
    model: torch.nn.Module
    opt: torch.optim.optimizer.Optimizer
    train_dloader: DataLoader
    val_dloader: DataLoader
    distance_fun: torch.nn.Module
    run_path: Path
    epochs: int

    val_steps: Optional[int] = None

    def train_step(self, batch):
        self.opt.zero_grad()
        pred_output = self.pred_calc_loss(batch)
        pred_output['loss'].backward()
        self.opt.step()
        return pred_output

    def pred_calc_loss(self, batch):
        X_supp, y_supp = batch['train']
        X_query, y_query = batch['test']
        pred_output = self.model(X_supp, y_supp, X_query)
        loss = self.calc_loss(pred_output['centroids'], pred_output['embeddings_query'], y_query)
        pred_output['loss'] = loss
        pred_output['batch'] = batch
        return pred_output

    def calc_loss(self, centroids, embeddings_query, y_query):
        """
        Calculate loss as specified in "Prototypical Networks for Few-shot Learning" page 3 algorithm 1
        using tensor broadcast operations.
        :param centroids: Centroids calculated from support set. Shape: [batch_size, num_classes, num_embedding_features]
        :param embeddings_query: Embeddings from images of query set. Shape: [batch_size, query_size * num_classes, num_embedding_features]
        :param y_query: Labels of query samples. Shape: [batch_size, query_size * num_classes]
        :return: the loss scalar value
        """
        centroids = centroids.unsqueeze(1)
        embeddings_query = embeddings_query.unsqueeze(2)
        loss_matrix = self.distance_fun(centroids, embeddings_query).sum(
            dim=-1)  # [batch_size, query_size * num_classes, num_classes]
        num_classes = centroids.shape[1]
        is_different_class = torch.arange(num_classes).view(1, 1, num_classes)
        is_different_class = y_query.unsqueeze(-1) != is_different_class
        loss_matrix[is_different_class] = (loss_matrix[is_different_class] * -1).logsumexp(dim=-1)
        num_classes_queries = y_query.shape[1]
        loss_value = loss_matrix.sum() / num_classes_queries
        return loss_value

    def setup_training(self):
        trainer = Engine(lambda e, b: self.train_step(b))

        RunningAverage(output_transform=lambda o: o['loss']).attach(trainer, 'running_avg_loss')
        Average(lambda o: o['loss']).attach(trainer, 'avg_loss')
        ProgressBar().attach(trainer, 'running_avg_loss')

        return trainer

    def train(self):
        trainer = self.setup_training()

        trainer.run(self.train_dloader, self.epochs)

    def validate(self):
        validator = Engine(lambda e, b: self.pred_calc_loss(b))
        self.model.get_probabilities = True

        def get_y_pred_y(o: dict):
            """
            :param o: output of forward method
            :return: tuple (y_pred, y)
            """
            return o['prob_query'].argmax(1).flatten(), o['batch']['y_query'].flatten()
        RunningAverage(output_transform=lambda o: o['loss']).attach(validator, 'running_avg_loss')
        ProgressBar().attach(validator, 'running_avg_loss')

        Accuracy(output_transform=get_y_pred_y).attach(validator, 'accuracy')
        results_train = validator.run(self.train_dloader, max_epochs=1, epoch_length=self.val_steps)
        results_val = validator.run(self.val_dloader, max_epochs=1, epoch_length=self.val_steps)
        print('Train accuracy:', results_train.metrics['accuracy'])
        print('Validation accuracy:', results_val.metrics['accuracy'])



def main():
    args = parse_args()
    exp_path = new_experiment_path()
    with open(exp_path / 'config.yaml', 'w') as f:
        yaml.dump(dict(args), f)


if __name__ == '__main__':
    main()
