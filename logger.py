from ignite.engine import Engine, Events
from path import Path
from torch.utils.tensorboard import SummaryWriter


def setup_logger(experiment_path: Path, engine: Engine):
    logger = SummaryWriter(experiment_path)

    @engine.on(Events.ITERATION_COMPLETED(every=1000))
    def log_scalars(engine):
        output = engine.staate.output
        logger.add_scalar('loss', output['loss'], engine.state.output)