from ignite.engine import Engine, Events
from path import Path
from torch.utils.tensorboard import SummaryWriter
import torch


def setup_logger(experiment_path: Path, engine: Engine, model: torch.nn.Module, steps_scalar: int = 100):
    logger = SummaryWriter(experiment_path)

    @engine.on(Events.ITERATION_COMPLETED(every=steps_scalar))
    def log_scalars(engine):
        output = engine.staate.output
        logger.add_scalar('loss_step', output['loss'], engine.state.output)
        norm_1_params = sum(p.norm(1) for p in model.parameters())
        norm_1_grads = sum(p.grad.norm(1) for p in model.parameters() if p.grad is not None)
        logger.add_scalar('norm_1_params', norm_1_params, engine.state.iteration)
        logger.add_scalar('norm_1_grad', norm_1_grads, engine.state.iteration)

    @engine.on("EVAL_DONE")
    def log_eval_reuslts(engine):
        eval_results = engine.state['eval_results']
        for split in ['train', 'val']:
            accuracy_split = eval_results[split].metrics['accuracy']
            loss_split = eval_results[split].metrics['avg_loss']
            logger.add_scalar(f'accuracy/{split}', accuracy_split, engine.state.iteration)
            logger.add_scalar(f'{split}_accuracy', accuracy_split, engine.state.iteration)
            logger.add_scalar(f'loss/{split}', loss_split, engine.state.iteration)
            logger.add_scalar(f'{split}_loss', loss_split, engine.state.iteration)