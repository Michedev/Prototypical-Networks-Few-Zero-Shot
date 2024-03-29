import timeit
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch.cuda.amp import autocast
from torch.nn.functional import one_hot
from typing import Literal
from ignite.contrib.handlers import ProgressBar
import tensorguard as tg
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
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

    opt: Optional[torch.optim.Optimizer]
    device: Union[str, torch.device]
    use_lr_decay: bool = False
    use_early_stop: bool = True
    early_stop_delta: float = 0.0
    early_stop_patience: int = 2
    lr_decay_gamma: float = None
    lr_decay_steps: int = None
    eval_steps: Optional[int] = None
    epoch_steps: int = field(init=True, default=200)
    zero_shot: bool = False
    batch_size: int = None
    early_stop_metric: Literal['loss', 'accuracy'] = 'accuracy'

    def __post_init__(self):
        assert self.early_stop_metric in ['loss', 'accuracy']
        self.model = self.model.to(self.device)
        if self.use_lr_decay:
            self.lr_decay = torch.optim.lr_scheduler.StepLR(self.opt, self.lr_decay_steps, self.lr_decay_gamma)

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
        X_query, y_query = batch['test']
        X_query = X_query.to(self.device)
        y_query = y_query.to(self.device)
        if self.zero_shot:
            classes_metadata = batch['meta']
            classes_metadata = classes_metadata.to(self.device)
            pred_output = self.model(classes_metadata, X_query)
        else:
            X_supp, y_supp = batch['train']
            X_supp = X_supp.to(self.device)
            y_supp = y_supp.to(self.device)
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
        num_classes = centroids.shape[1]
        centroids = centroids.unsqueeze(1)
        embeddings_query = embeddings_query.unsqueeze(2)
        loss_matrix = self.distance_fun(centroids, embeddings_query)  # [batch_size, query_size, num_classes]
        tg.guard(loss_matrix, "*, QUERY_SIZE, NUM_CLASSES")
        index_correct_class = torch.arange(num_classes, device=y_query.device).view(1, 1, num_classes)
        index_correct_class = y_query.unsqueeze(-1) == index_correct_class
        index_correct_class = index_correct_class\
            .expand(index_correct_class.shape[0], index_correct_class.shape[1], num_classes)
        index_correct_class = index_correct_class.int()
#         tg.guard(index_correct_class, "*, QUERY_SIZE, NUM_CLASSES")
#         tg.guard(loss_matrix, "*, QUERY_SIZE, NUM_CLASSES")

        loss_value = (loss_matrix * -1).logsumexp(dim=-1).sum()
        loss_value += (loss_matrix * index_correct_class).sum()
        num_classes_queries = y_query.shape[1]
        loss_value /= num_classes_queries
        return loss_value

    def setup_training(self):
        assert self.batch_size is not None
        trainer = Engine(lambda e, b: self.train_step(b))
        trainer.register_events("EVAL_DONE")
        Average(lambda o: o['loss']).attach(trainer, 'avg_loss')
        state_vars = dict(model=self.model, opt=self.opt, trainer=trainer)
        checkpoint_handler = ModelCheckpoint(self.run_path, '', score_function=lambda e: e.state.metrics['val_accuracy'],
                                             score_name='val_accuracy', n_saved=2, global_step_transform=lambda e, evt_name: e.state.epoch)
        if checkpoint_handler.last_checkpoint:
            checkpoint_handler.load_objects(state_vars, self.run_path / checkpoint_handler.last_checkpoint)
        trainer.add_event_handler("EVAL_DONE", lambda e: checkpoint_handler(e, state_vars))
        if self.use_lr_decay:
            trainer.add_event_handler(Events.ITERATION_COMPLETED, lambda e: self.lr_decay.step(e.state.iteration * self.batch_size))

        RunningAverage(output_transform=lambda o: o['loss']).attach(trainer, 'running_avg_loss')
        ProgressBar().attach(trainer, ['running_avg_loss'])
        logger.setup_logger(self.run_path, trainer, self.model)

        @trainer.on(Events.EPOCH_COMPLETED)
        def eval_and_log(e: Engine):
            eval_results = self.eval()
            e.state.metrics['val_accuracy'] = eval_results['val'].metrics['accuracy'] 
            e.state.metrics['val_loss'] = eval_results['val'].metrics['avg_loss']
            e.state.eval_results = eval_results
            e.fire_event("EVAL_DONE")

        if self.use_early_stop:
            es = self.make_early_stopper(trainer)
            trainer.add_event_handler("EVAL_DONE", es)

        return trainer
    
    def make_early_stopper(self, trainer):
        if self.early_stop_metric == 'loss':
            key_name = 'val_loss'
            c = -1
        else:
            c = 1
            key_name = 'val_accuracy'
        return EarlyStopping(self.early_stop_patience, lambda e: c * e.state.metrics[key_name],
                               trainer, min_delta=self.early_stop_delta)


    def train(self):
        trainer = self.setup_training()
        print('starting training...')
        trainer.run(self.train_dloader, self.train_epochs, self.epoch_steps)

    def eval(self):
        """
        Predict and calculate metrics on both train dataloader and validation dataloader.
            (train dataloader is optional for this method)
        :return: Evaluation dict - Relevant features are ['train', 'val'] which are
                 ignite.State who contains: <br>
                 - result['train'].metrics['accuracy'] <br>
                 - result['train'].metrics['avg_loss'] <br>
                 - result['val'].metrics['accuracy'] <br>
                 - result['val'].metrics['avg_loss']

        :rtype: dict
        """
        validator = self.setup_evaluator()
        result = dict()
        self.model.eval()
        with torch.no_grad():
            if self.train_dloader is not None:
                start = timeit.default_timer()
                results_train = validator.run(self.train_dloader, max_epochs=1, epoch_length=self.eval_steps)
                end = timeit.default_timer()
                results_train.eval_duration_seconds = end - start
                result['train'] = results_train
            tg.reset()
            start = timeit.default_timer()
            results_val = validator.run(self.val_dloader, max_epochs=1, epoch_length=self.eval_steps)
            end = timeit.default_timer()
            tg.reset()
            results_val.eval_duration_seconds = end - start
            result['val'] = results_val
        self.model.train()
        if self.train_dloader is not None:
            print('Train accuracy:', results_train.metrics['accuracy'])
        print('Test accuracy:', results_val.metrics['accuracy'])
        return result

    def setup_evaluator(self):
        validator = Engine(lambda e, b: self.pred_calc_loss(b))
        self.model.get_probabilities = True

        def get_y_pred_y(o: dict):
            """
            :param o: output of forward method
            :return: tuple (y_pred, y) both with shape [batch_size * query_size, num_classes] in OHE
            """
            y_pred, y = o['prob_query'].argmax(1).flatten(), o['batch']['test'][1].flatten()
            num_classes = int(y.max().item())+1
            y_pred = one_hot(y_pred, num_classes)
            y = one_hot(y, num_classes)
            assert y_pred.shape == y.shape, f"{y_pred.shape} != {y.shape}"
            return y_pred, y

        RunningAverage(output_transform=lambda o: o['loss']).attach(validator, 'running_avg_loss')
        Average(lambda o: o['loss']).attach(validator, 'avg_loss')
        ProgressBar().attach(validator, ['running_avg_loss'])
        Accuracy(output_transform=get_y_pred_y, is_multilabel=True).attach(validator, 'accuracy')
        return validator
