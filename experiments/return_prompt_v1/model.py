"""Per-AP DT with an additional scalar return prompt at every timestep."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(),
                                 nn.Linear(4*dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, length, dim = x.shape
        def split(layer):
            return layer(x).view(b, length, self.heads, dim//self.heads).transpose(1, 2)
        z = F.scaled_dot_product_attention(split(self.q), split(self.k), split(self.v),
                dropout_p=self.dropout if self.training else 0., is_causal=True)
        z = z.transpose(1, 2).contiguous().view(b, length, dim)
        x = self.ln1(x + self.proj_drop(self.proj(z)))
        return self.ln2(x + self.mlp(x))


class ReturnPromptDT(nn.Module):
    def __init__(self, state_dim=110, h_dim=128, n_blocks=3, n_heads=1,
                 drop_p=.1, max_timestep=10080):
        super().__init__()
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = nn.Linear(1, h_dim)
        self.embed_prompt = nn.Linear(1, h_dim)
        self.embed_state = nn.Linear(state_dim, h_dim)
        self.embed_ant = nn.Embedding(3, h_dim)
        self.embed_sleep = nn.Embedding(4, h_dim)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.blocks = nn.Sequential(*[Block(h_dim, n_heads, drop_p) for _ in range(n_blocks)])
        self.ant_head = nn.Linear(h_dim, 3)
        self.sleep_head = nn.Linear(h_dim, 4)

    def forward(self, timesteps, states, actions, returns_to_go, prompts):
        b, t, _ = states.shape
        if prompts.shape != (b, t, 1):
            raise ValueError('return prompts must have shape [batch, time, 1]')
        te = self.embed_timestep(timesteps)
        tokens = torch.stack((self.embed_rtg(returns_to_go)+te,
                              self.embed_prompt(prompts)+te,
                              self.embed_state(states)+te,
                              self.embed_ant(actions[..., 0].long())+
                              self.embed_sleep(actions[..., 1].long())+te), dim=2)
        h = self.blocks(self.embed_ln(tokens.reshape(b, 4*t, -1)))
        state_h = h.reshape(b, t, 4, -1)[:, :, 2]
        return self.ant_head(state_h), self.sleep_head(state_h)


def prompt_from_rtg(raw_rtg, timesteps, mu, sigma):
    """Training-supported known-task mapping. Never fit these arrays on eval data."""
    return torch.sigmoid((raw_rtg-mu[timesteps].unsqueeze(-1))/sigma[timesteps].unsqueeze(-1))
