# Weight-normalized PDT, B scene, policy pc1.8 (prompt 1.8/3.3 for all APs). Re-run with finish_packet hook logging every stringent
# packet + all AP states at each drop. Cases: M (mean RTG) seeds 1-20 (1-5 re-check vs PDT_single200_uniformprompt_RTG_v1), AP (per-AP RTG) seeds 6-20.
import os, sys, json, importlib.util, concurrent.futures, subprocess, shutil
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); B = Path(__file__).parent; sys.path.insert(0, str(R))
spec = importlib.util.spec_from_file_location('base', R / 'experiments/PDT_single200_uniformprompt_RTG_v1/runner.py'); base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
O = R / 'analysis/sim_stats/PDT_pc18_seed20_events_v1'; base.O = O
TAG = 'pc1.8_trainseed5_cpfinal'
CASES = [dict(env='B', tag=TAG, pc=1.8, seed=s, var='M') for s in range(1, 21)] + [dict(env='B', tag=TAG, pc=1.8, seed=s, var='AP') for s in range(6, 21)]
base.CASES = CASES
def run_job(i):
    from network.user_equipment import UserEquipment
    EV = []; orig = UserEquipment.finish_packet
    def snap(net):
        return [dict(id=b.id, sleep=int(b.sleep), next_sleep=int(b._next_sleep), ants=int(b.num_ant), wake_left_ms=round(1000 * float(b.wakeup_time), 1),
                     serving=len(b.ues), queue=len(b.queue), covered=len(b.covered_ues)) for b in net.bss.values()]
    def fp(self, success):
        if self.service == 0:
            EV.append(dict(step=int(round(self.net._time / 0.02 - 0.5)), ue=str(self.id), success=bool(success), delay_ms=round(1000 * float(self.delay), 1),
                           serving=[b.id for b in self.bss], cover=sorted(b.id for b in self._cover_bss), pkt_index=50 - int(self.packets_remaining),
                           aps=None if success else snap(self.net)))
        return orig(self, success)
    UserEquipment.finish_packet = fp
    base.worker(i); c = CASES[i]
    d = O / c['var'] / c['tag'] / 'B' / 'B' / f"SEED{c['seed']}"; (d / 'events.json').write_text(json.dumps(EV))
    print('DONE', i, c['var'], c['seed'], 'str_pkts', len(EV), 'drops', sum(not e['success'] for e in EV), flush=True)
def main():
    assert not O.exists() and not (B / 'plan.json').exists() and base.sha(base.M) == base.H
    (B / 'logs').mkdir(exist_ok=True)
    (B / 'plan.json').write_text(json.dumps(dict(model=str(base.M), sha256=base.H, cases=CASES, runs=len(CASES), parallel=12), indent=1))
    def job(i):
        assert shutil.disk_usage(R).free > 0.8 * 1024**3
        with (B / 'logs' / f'case-{i}.log').open('x') as f: rc = subprocess.call([sys.executable, '-u', __file__, 'job', str(i)], cwd=R, stdout=f, stderr=subprocess.STDOUT)
        assert rc == 0, i; return i
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as p:
        for i in p.map(job, range(len(CASES))):
            done.append(i); (B / 'state.json').write_text(json.dumps(dict(complete=len(done), total=len(CASES)))); print('COMPLETE', i, len(done), flush=True)
    (B / 'complete.json').write_text(json.dumps(dict(runs=len(CASES), model_sha256=base.H))); print('ALL_COMPLETE', flush=True)
if __name__ == '__main__':
    if len(sys.argv) > 1: run_job(int(sys.argv[2]))
    else: main()
