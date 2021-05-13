from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
RUN = ROOT / 'run'
if not RUN.exists():
    RUN.mkdir()


def new_experiment_path():
    i = 0
    while (RUN / str(i)).exists(): i += 1
    result = (RUN / str(i))
    result.mkdir()
    return result
