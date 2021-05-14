from dataclasses import dataclass
from typing import Optional, Union

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage, Average, Accuracy
from path import Path
from torch.utils.data import DataLoader

import logger


@dataclass
class Trainer:
    model: torch.nn.Module
    train_dloader: DataLoader
    val_dloader: DataLoader
    distance_fun: torch.nn.Module
    run_path: Path
    train_epochs: int

    opt: Optional[torch.optim.optimizer.Optimizer]
    device: str
    val_steps: Optional[int] = None

    def __post_init__(self):
        self.model.to(self.device)
        self.lr_decay = torch.optim.lr_scheduler.StepLR(self.opt, 3_000, 0.5)

    def train_step(self, batch):
        self.opt.zero_grad()
        pred_output = self.pred_calc_loss(batch)
        pred_output['loss'].backward()
        self.opt.step()
        return pred_output

    def pred_calc_loss(self, batch):
        """
        Predict then calculate loss
        :param batch:
        :type batch:
        :return:
        :rtype:
        """
        for k in batch:
            batch[k].to(self.device)
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
        state_vars = dict(model=self.model, opt=self.opt, trainer=trainer)
        checkpoint_handler = ModelCheckpoint(self.run_path, 'checkpoint', score_name='avg_loss', n_saved=2,
                                             global_step_transform=lambda: trainer.state.epoch)
        if checkpoint_handler.last_checkpoint:
            checkpoint_handler.load_objects(state_vars, self.run_path / checkpoint_handler.last_checkpoint)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda e: checkpoint_handler(e, state_vars))
        trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda e: self.lr_decay.step(e.state.iteration))

        RunningAverage(output_transform=lambda o: o['loss']).attach(trainer, 'running_avg_loss')
        Average(lambda o: o['loss']).attach(trainer, 'avg_loss')
        ProgressBar().attach(trainer, 'running_avg_loss')
        logger.setup_logger(self.run_path, trainer, self.model)

        @trainer.on(Events.EPOCH_COMPLETED(every=5))
        def eval_and_log(e: Engine):
            eval_results = self.eval()
            e.state['eval_results'] = eval_results
            e.fire_event("EVAL_DONE")

        return trainer

    def train(self):
        trainer = self.setup_training()

        trainer.run(self.train_dloader, self.train_epochs)

    def eval(self):
        """
        Predict and calculate metrics on both train dataloader and validation dataloader. (train dataloader is optional)
        :return:
        :rtype:
        """
        validator = self.setup_evaluator()
        result = dict()
        if self.train_dloader is not None:
            results_train = validator.run(self.train_dloader, max_epochs=1, epoch_length=self.val_steps)
            result['train'] = results_train
        results_val = validator.run(self.val_dloader, max_epochs=1, epoch_length=self.val_steps)
        result['val'] = results_val
        if self.train_dloader is not None:
            print('Train accuracy:', results_train.metrics['accuracy'])
        print('Validation accuracy:', results_val.metrics['accuracy'])
        return result

    def setup_evaluator(self):
        validator = Engine(lambda e, b: self.pred_calc_loss(b))
        self.model.get_probabilities = True

        def get_y_pred_y(o: dict):
            """
            :param o: output of forward method
            :return: tuple (y_pred, y)
            """
            return o['prob_query'].argmax(1).flatten(), o['batch']['y_query'].flatten()

        RunningAverage(output_transform=lambda o: o['loss']).attach(validator, 'running_avg_loss')
        Average(lambda o: o['loss']).attach(validator, 'avg_loss')
        ProgressBar().attach(validator, 'running_avg_loss')
        Accuracy(output_transform=get_y_pred_y, is_multilabel=True).attach(validator, 'accuracy')
        return validator