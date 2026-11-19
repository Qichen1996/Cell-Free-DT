"""Run unchanged project trainer in isolated processes, timing loading separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/workspace/Cell-Free-DT-new')
OUTPUT = ROOT / 'experiments/trajectory_train_timing_20261018'

def child(mode, label):
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import torch
    import trajectory_io
    import train_dt
    from types import SimpleNamespace
    folder = OUTPUT / label
    folder.mkdir(exist_ok=False)
    timings = {}
    original_load = trajectory_io.load_trajectory_pickle
    def timed_load(path):
        start = time.perf_counter()
        value = original_load(path)
        timings['pickle_load_seconds'] = time.perf_counter() - start
        return value
    trajectory_io.load_trajectory_pickle = timed_load
    original_dataset = train_dt.TrajectoryDataset
    def timed_dataset(*args, **kwargs):
        start = time.perf_counter()
        value = original_dataset(*args, **kwargs)
        timings['dataset_seconds'] = time.perf_counter() - start
        return value
    train_dt.TrajectoryDataset = timed_dataset
    original_step = torch.optim.AdamW.step
    original_zero = torch.optim.AdamW.zero_grad
    def timed_zero(self, *args, **kwargs):
        if 'first_update_start' not in timings:
            torch.cuda.synchronize()
            timings['first_update_start'] = time.perf_counter()
        return original_zero(self, *args, **kwargs)
    def timed_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        torch.cuda.synchronize()
        timings['last_update_end'] = time.perf_counter()
        return result
    torch.optim.AdamW.zero_grad = timed_zero
    torch.optim.AdamW.step = timed_step
    path = ROOT / 'analysis/trajectory/B1000_multicp_70/B.pkl'
    if mode == 'zstd':
        path = Path(str(path) + '.zst')
    args = SimpleNamespace(rtg_scale=1, norm=False, batch_size=256, lr=5e-4,
        wt_decay=1e-4, max_train_iters=2, num_updates_per_iter=150, context_len=8,
        n_blocks=3, embed_dim=128, n_heads=1, dropout_p=.1, seed=1,
        dataset_files=[str(path)], dataset_dir='', log_dir=str(folder), device='cuda:0',
        seed_fraction=1.0, single_ap=True, deployment_prompt=True, no_prompt=False,
        num_workers=4)
    torch.cuda.init()
    start = time.perf_counter()
    train_dt.train(args, None)
    torch.cuda.synchronize()
    timings['trainer_total_seconds'] = time.perf_counter() - start
    timings['update_interval_seconds'] = timings.pop('last_update_end') - timings.pop('first_update_start')
    timings['excluding_dataset_seconds'] = timings['trainer_total_seconds'] - timings['dataset_seconds']
    timings.update(mode=mode, label=label, updates=300, workers=4, dataset=str(path),
                   note='Warm/uncontrolled OS cache; unchanged train_dt.train, synchronized optimizer timing includes first inter-round checkpoint save.')
    checkpoint = torch.load(folder / 'dt_8_1.pt', map_location='cpu', weights_only=False)
    digest = hashlib.sha256()
    for key, value in sorted(checkpoint['model_state_dict'].items()):
        digest.update(key.encode())
        digest.update(value.numpy().tobytes())
    timings['model_tensor_sha256'] = digest.hexdigest()
    (folder / 'timing.json').write_text(json.dumps(timings, indent=2))
    print('TRIAL_COMPLETE', json.dumps(timings), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--label')
    args = parser.parse_args()
    if args.mode:
        child(args.mode, args.label)
        return
    OUTPUT.mkdir(exist_ok=False)
    results = []
    for index, mode in enumerate(['raw', 'zstd', 'zstd', 'raw']):
        label = f'{index+1}_{mode}'
        subprocess.run([sys.executable, __file__, '--mode', mode, '--label', label], check=True)
        results.append(json.loads((OUTPUT / label / 'timing.json').read_text()))
    (OUTPUT / 'summary.json').write_text(json.dumps(results, indent=2))
    print('ALL_TIMING_COMPLETE', json.dumps(results), flush=True)

if __name__ == '__main__':
    main()
