"""User-authorized continuation below -576; reuse the verified DT evaluator."""
import importlib.util,json,hashlib,sys,shutil
from pathlib import Path
ROOT=Path('/workspace/Cell-Free-DT-new')
OLD=ROOT/'experiments/selected_seven_pc33_DT_lower_B'
BASE=ROOT/'experiments/selected_seven_pc33_DT_lower2_B'
spec=importlib.util.spec_from_file_location('verified_lower',OLD/'runner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
prior=json.loads((OLD/'plan.json').read_text())
assert hashlib.sha256(module.SOURCE.read_bytes()).hexdigest()==prior['source_sha256']
assert hashlib.sha256(Path(prior['model']).read_bytes()).hexdigest()==module.HASH==prior['model_sha256']
assert shutil.disk_usage(ROOT).free>4*1024**3
module.BASE=BASE
module.STATS=ROOT/'analysis/sim_stats/selected_seven_pc33_DT_lower2_B'
module.TARGETS=[-606.0014544159304,-636.0014544159304,-666.0014544159304]
module.__file__=__file__
if __name__=='__main__':
 module.main()
 if len(sys.argv)==1:print('ALL_COMPLETE_DT_LOWER2_RTG_B_15',flush=True)
