import bisect
import hashlib
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class ReturnPromptDataset(Dataset):
    """Original physical 70% datasets; no second seed-fraction truncation."""
    def __init__(self, paths, context=8, bootstrap=256, seed=1):
        self.context = context
        self.trajectories = []
        self.ends = []
        self.counts = []
        self.sources = []
        moments = []
        moments2 = []
        for path in paths:
            path = Path(path)
            with path.open('rb') as f:
                items = pickle.load(f)
            print('DATA_SELECTION', str(path), len(items), 'all retained', flush=True)
            self.sources.append({'path': str(path), 'trajectories': len(items),
                                 'bytes': path.stat().st_size})
            for tr in items:
                obs, act, rew = (np.asarray(tr[k]) for k in ('observations','actions','rewards'))
                assert obs.shape[0] == 10080 and obs.shape[-1] == 110
                assert act.shape == (*obs.shape[:2], 2)
                if rew.ndim == 2:
                    rew = rew[..., None]
                assert rew.shape == (*obs.shape[:2], 1)
                assert all(np.isfinite(v).all() for v in (obs, act, rew))
                assert np.all(act == act.astype(np.int64))
                assert np.all((act[...,0]>=0)&(act[...,0]<3))
                assert np.all((act[...,1]>=0)&(act[...,1]<4))
                # Preserve the exact raw, gamma=1 RTG convention of the old dataset.
                rtg = np.cumsum(rew[::-1], axis=0, dtype=np.float32)[::-1].copy()
                moments.append(rtg.astype(np.float64).mean(axis=1)[:,0])
                moments2.append(np.square(rtg.astype(np.float64)).mean(axis=1)[:,0])
                self.trajectories.append({'observations':obs,'actions':act,'rtg':rtg})
                windows = len(obs)-context+1
                self.counts.append(windows)
                self.ends.append((self.ends[-1] if self.ends else 0)+windows*obs.shape[1])
        # Each bootstrap replicate draws whole trajectories with replacement.
        # Each trajectory has equal weight; APs are equally weighted within it.
        # Pooled across all seven datasets, not per-task normalization.
        m, m2 = np.stack(moments), np.stack(moments2)
        rng = np.random.default_rng(seed)
        n = len(m)
        weights = rng.multinomial(n, np.full(n, 1/n), size=bootstrap)/n
        means, seconds = weights @ m, weights @ m2
        variances = np.maximum(seconds-np.square(means), 0.)
        self.mu = means.mean(axis=0).astype(np.float32)
        self.sigma = np.maximum(np.sqrt(variances.mean(axis=0)), 1e-6).astype(np.float32)
        # sigma is outcome dispersion, NOT the standard error of bootstrap means.
        self.prompt_max = 0.
        self.prompt_min = 1.
        for tr in self.trajectories:
            z=(tr['rtg']-self.mu[:,None,None])/self.sigma[:,None,None]
            tr['return_prompt']=(1/(1+np.exp(-np.clip(z,-60,60)))).astype(np.float32)
            self.prompt_max=max(self.prompt_max,float(tr['return_prompt'].max()))
            self.prompt_min=min(self.prompt_min,float(tr['return_prompt'].min()))
        self.metadata={'bootstrap_replicates':bootstrap,'bootstrap_seed':seed,
          'resampling_unit':'whole trajectory; uniform trajectory weights; uniform APs within trajectory',
          'pooling':'all seven training datasets together, independently at each absolute timestep',
          'sigma_definition':'sqrt(mean bootstrap population variances), not bootstrap mean standard error',
          'sigma_floor':1e-6,'rtg_scale':1,'gamma':1,'trajectories':n,
          'prompt_min':self.prompt_min,'prompt_max':self.prompt_max,'sources':self.sources}
        print('RETURN_PROMPT_STATS',self.metadata,flush=True)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, idx):
        ti=bisect.bisect_right(self.ends,idx)
        offset=idx-(self.ends[ti-1] if ti else 0)
        ap,start=divmod(offset,self.counts[ti])
        end=start+self.context
        tr=self.trajectories[ti]
        return (torch.arange(start,end),
                torch.from_numpy(tr['observations'][start:end,ap]),
                torch.from_numpy(tr['actions'][start:end,ap]),
                torch.from_numpy(tr['rtg'][start:end,ap]),
                torch.from_numpy(tr['return_prompt'][start:end,ap]))
