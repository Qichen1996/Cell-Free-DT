"""Load trusted local pickle or Zstd pickle directly into memory."""
import pickle
from pathlib import Path

def load_trajectory_pickle(path):
    path=Path(path)
    # Preserve explicit existing paths. Fallback only when the original was retired.
    if not path.exists() and path.suffix=='.pkl' and Path(str(path)+'.zst').is_file():
        path=Path(str(path)+'.zst')
    with path.open('rb') as f:
        if path.suffix=='.zst':
            try:import zstandard
            except ImportError as e:raise ImportError('Zstd trajectory requires the zstandard package') from e
            with zstandard.ZstdDecompressor().stream_reader(f) as reader:
                return pickle.load(reader)
        return pickle.load(f)
