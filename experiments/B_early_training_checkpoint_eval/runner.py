"""Run the already-authorized B subset before A training finishes."""
from pathlib import Path
source = Path('/workspace/Cell-Free-DT-new/experiments/B_early_training_checkpoint_eval/original_runner.py').read_text()
source = source.replace("BASE=ROOT/'experiments/AB_new_training_checkpoint_eval'", "BASE=ROOT/'experiments/B_early_training_checkpoint_eval'")
source = source.replace('range(35)', 'range(10)').replace('175', '50').replace('max_workers=3', 'max_workers=2').replace('parallel=3', 'parallel=2')
source = source.replace("for name in ['A_pc_sweep_12_16','B_pc18_seeds23']:", "for name in ['B_pc18_seeds23']:")
scope = {'__name__': 'early_b_module', '__file__': __file__}
exec(compile(source, __file__, 'exec'), scope)
original_specs = scope['specs']
scope['specs'] = lambda: [s for s in original_specs() if s['scenario'] == 'B']
import sys
if len(sys.argv) > 1:
    scope['worker'](int(sys.argv[2]), int(sys.argv[3]), sys.argv[1] == 'preflight')
else:
    scope['main']()
