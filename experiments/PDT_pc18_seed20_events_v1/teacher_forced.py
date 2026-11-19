# Offline, read-only: feed PPO training trajectories (B 10 policies; A, C) into weight-normalized PDT with their training prompt,
# per-AP return-to-go from the trajectory; compare argmax action with PPO action at each (AP, step). dim0=antenna switch, dim1=sleep mode.
import sys, json, numpy as np, torch
from pathlib import Path
R = Path('/workspace/Cell-Free-DT-new'); sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'experiments/selected_seven_pc33_v1'))
import train_mixed as training
from trajectory_io import load_trajectory_pickle as L
torch.set_num_threads(16)
M = R / 'dt_results/selected_seven_Bsingle200_v1/PDT/model.pt'
model = training.Model(True); ck = torch.load(M, map_location='cpu', weights_only=False); model.load_state_dict(ck['model_state_dict'], strict=True); model.eval()
MAN = json.loads((R / 'experiments/tradeoff_selected_single_collect200_v1/dataset-manifest.json').read_text())['files']
@torch.no_grad()
def run(tr, p):
    obs = torch.as_tensor(tr['observations'], dtype=torch.float32).permute(1, 0, 2); act = torch.as_tensor(tr['actions'], dtype=torch.float32).permute(1, 0, 2)
    rew = torch.as_tensor(tr['rewards'][:, :, 0], dtype=torch.float32).T; rtg = rew.flip(1).cumsum(1).flip(1).unsqueeze(-1)
    n, T, _ = obs.shape; ts = torch.arange(T).repeat(n, 1); pred = torch.zeros(n, T, 2, dtype=torch.long)
    steps = list(range(7, T))
    for c in range(0, len(steps), 512):
        ks = steps[c:c + 512]; idx = torch.stack([torch.arange(k - 7, k + 1) for k in ks])  # (b,8)
        S = obs[:, idx].reshape(-1, 8, 110); A = act[:, idx].reshape(-1, 8, 2); G = rtg[:, idx].reshape(-1, 8, 1); TT = ts[:, idx].reshape(-1, 8)
        h = model(TT, S, A, G, torch.full((S.shape[0], 1), p))
        for d in range(2): pred[:, ks, d] = h[d][:, -1].argmax(-1).reshape(n, len(ks))
    a = act[:, 7:].long(); q = pred[:, 7:]
    ppo_s, pdt_s = a[..., 1], q[..., 1]
    return dict(agree_ant=float((a[..., 0] == q[..., 0]).float().mean()), agree_sleep=float((ppo_s == pdt_s).float().mean()),
                ppo_sleep_hist=np.bincount(ppo_s.flatten(), minlength=4).tolist(), pdt_sleep_hist=np.bincount(pdt_s.flatten(), minlength=4).tolist(),
                pdt_deeper=float((pdt_s > ppo_s).float().mean()), pdt_shallower=float((pdt_s < ppo_s).float().mean()),
                ppo0_pdt_ge2=float(((ppo_s == 0) & (pdt_s >= 2)).float().sum() / max(1, (ppo_s == 0).sum())))
out = {}
tags = sorted({f['model']['tag'] for f in MAN}, key=lambda t: float(t[2:5]))
for t in tags:
    pc = float(t[2:5])
    for s in [1, 6]:
        f = next(f for f in MAN if f['model']['tag'] == t and f['seed'] == s); tr = L(f['path'])[0]
        prompts = [pc / 3.3] + ([1.5 / 3.3, 2.1 / 3.3] if pc == 1.8 else [])
        for p in prompts:
            r = run(tr, p); out[f'{t}|seed{s}|prompt{p:.3f}'] = r; print(t, s, round(p, 3), json.dumps(r), flush=True)
for e in ['A', 'C']:
    tr = L(R / f'analysis/trajectory/uniform/{e}.pkl.zst')[0]; r = run(tr, 1.8 / 3.3); out[f'{e}|seed1|prompt0.545'] = r; print(e, json.dumps(r), flush=True)
(Path(__file__).parent / 'teacher_forced.json').write_text(json.dumps(out, indent=1)); print('TF_DONE')
