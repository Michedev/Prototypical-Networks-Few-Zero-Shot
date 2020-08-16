from path import Path

ROOT = Path(__file__).parent
DATAFOLDER = ROOT / 'data'
OMNIGLOTFOLDER = DATAFOLDER / 'omniglot-py'
MINIIMAGENETFOLDER = DATAFOLDER / 'miniimagenet'
WEIGHTSFOLDER = ROOT / 'model_weights'
LOGFOLDER = ROOT / 'logs'
if not WEIGHTSFOLDER.exists():
    WEIGHTSFOLDER.mkdir()
if not LOGFOLDER.exists():
    LOGFOLDER.mkdir()
EMBEDDING_PATH = WEIGHTSFOLDER / 'embedding.pth'
BEST_EMBEDDING_PATH = WEIGHTSFOLDER / 'best_embedding.pth'