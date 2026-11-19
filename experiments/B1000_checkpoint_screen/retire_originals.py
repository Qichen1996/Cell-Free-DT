"""Delete only eight byte-verified raw copies and four timing-test checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zstandard

PROJECT = Path('/workspace/Cell-Free-DT-new')
ROOT = PROJECT / 'analysis/trajectory'
FILES = ['B1000_multicp_70/B.pkl', 'B2000_e180_uniform/B.pkl',
         'B40-05-05-3/B36/B.pkl', 'B40-05-05-3/B4/B.pkl',
         'B500_e450_uniform/B.pkl', 'uniform/A.pkl', 'uniform/B.pkl', 'uniform/C.pkl']

def sha(stream):
    result = hashlib.sha256()
    size = 0
    while block := stream.read(8 * 1024**2):
        result.update(block)
        size += len(block)
    return result.hexdigest(), size

def regular(path, root):
    assert path.is_file() and not path.is_symlink(), str(path)
    assert path.resolve().is_relative_to(root.resolve()), str(path)
    assert all(not parent.is_symlink() for parent in path.parents), str(path)

def main():
    sys.path.insert(0, str(PROJECT))
    from trajectory_io import load_trajectory_pickle
    import numpy as np
    records = []
    for name, count in zip(FILES, [35, 35, 17, 140, 35, 35, 35, 35]):
        raw = ROOT / name
        compressed = Path(str(raw) + '.zst')
        evidence = Path(str(compressed) + '.verification.json')
        for path in [raw, compressed, evidence]:
            regular(path, ROOT)
        original_stat = raw.stat()
        with raw.open('rb') as stream:
            original_hash, size = sha(stream)
        with compressed.open('rb') as stream, zstandard.ZstdDecompressor().stream_reader(stream) as reader:
            assert sha(reader) == (original_hash, size)
        assert json.loads(evidence.read_text())['source_sha256'] == original_hash
        data = load_trajectory_pickle(compressed)
        assert len(data) == count
        for trajectory in data:
            for key in ['observations', 'actions', 'rewards']:
                assert len(trajectory[key]) == 10080 and np.isfinite(trajectory[key]).all()
        del data
        records.append(dict(raw=str(raw), compressed=str(compressed), source_bytes=size,
                            source_sha256=original_hash, mtime_ns=original_stat.st_mtime_ns))
        print('SAFE_TO_RETIRE', name, flush=True)
    scratch_root = PROJECT / 'experiments/trajectory_train_timing_20261018'
    scratch = [scratch_root / name / 'dt_8_1.pt' for name in ['1_raw', '2_zstd', '3_zstd', '4_raw']]
    summary = json.loads((scratch_root / 'summary.json').read_text())
    assert len(summary) == 4 and len({x['model_tensor_sha256'] for x in summary}) == 1
    for path in scratch:
        regular(path, scratch_root)
    manifest = PROJECT / 'experiments/trajectory_retirement_20261018.json'
    assert not manifest.exists()
    result = dict(verified=records, removed=[], temporary_models_removed=[],
                  free_before=shutil.disk_usage(ROOT).free)
    with manifest.open('x') as stream:
        json.dump(result, stream, indent=2)
    for item in records:
        raw = Path(item['raw'])
        assert raw.stat().st_size == item['source_bytes'] and raw.stat().st_mtime_ns == item['mtime_ns']
        raw.unlink()
        result['removed'].append(str(raw))
        data = load_trajectory_pickle(raw)
        assert data
        del data
        manifest.write_text(json.dumps(result, indent=2))
        print('REMOVED_RAW_FALLBACK_OK', str(raw), flush=True)
    for path in scratch:
        result['temporary_models_removed'].append(dict(path=str(path), size=path.stat().st_size))
        path.unlink()
    result['free_after'] = shutil.disk_usage(ROOT).free
    result['complete'] = True
    manifest.write_text(json.dumps(result, indent=2))
    print('CLEANUP_COMPLETE', json.dumps(dict(raw_files=len(records), temporary_models=len(scratch),
          raw_bytes=sum(x['source_bytes'] for x in records), free_bytes=result['free_after'])), flush=True)

if __name__ == '__main__':
    main()
