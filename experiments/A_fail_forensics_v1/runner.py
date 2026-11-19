# Forensics of high-drop PDT_APSTR_v2 evaluation runs on scenario A: re-run the exact same eval job (same model, prompt, seed)
# with a hook on UserEquipment.finish_packet logging every stringent packet + all AP states at each drop. Output in a separate dir.
import os, sys, json, importlib.util
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R))
os.environ.update(DEBUG='0', TRAIN='0', EVAL='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
spec = importlib.util.spec_from_file_location('ev2', R / 'experiments/PDT_APSTR_eval_v2/runner.py'); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
OUT = R / 'analysis/sim_stats/A_fail_forensics_v1'; ev.O = OUT; ev.B = Path(__file__).parent
from network.user_equipment import UserEquipment
EV = []; orig = UserEquipment.finish_packet
def snap(net):
    return [dict(id=b.id, sleep=int(b.sleep), next_sleep=int(b._next_sleep), ants=int(b.num_ant), wake_left_ms=round(1000 * float(b.wakeup_time), 1),
                 serving=len(b.ues), queue=len(b.queue), covered=len(b.covered_ues)) for b in net.bss.values()]
def fp(self, success):
    if self.service == 0:
        EV.append(dict(step=int(round(self.net._time / 0.02 - 0.5)), ue=str(self.id), success=bool(success), delay_ms=round(1000 * float(self.delay), 1),
                       pos=[round(float(x)) for x in self.pos[:2]], serving=[b.id for b in self.bss], cover=sorted(b.id for b in self._cover_bss),
                       pkt_index=50 - int(self.packets_remaining), aps=None if success else snap(self.net)))
    return orig(self, success)
UserEquipment.finish_packet = fp
i, seed = int(sys.argv[1]), int(sys.argv[2]); s = ev.S[i]; assert s['env'] == 'A'
ev.worker(i, seed, False)
d = OUT / s['tag'] / 'A' / 'A' / f'SEED{seed}'; (d / 'events.json').write_text(json.dumps(EV))
t = d / 'trajectory.csv'
if t.exists(): t.unlink()
print('DONE', s['tag'], seed, 'str_pkts', len(EV), 'drops', sum(not e['success'] for e in EV), flush=True)
