from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
RUN = ROOT / 'run'
RESULTS = ROOT / 'results'
if not RUN.exists():
    RUN.mkdir()
if not RESULTS.exists():
    RESULTS.mkdir()

def new_experiment_path():
    i = 0
    while (RUN / str(i)).exists(): i += 1
    result = (RUN / str(i))
    result.mkdir()
    return result
