import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.drop_p = drop_p
        self.proj_drop = nn.Dropout(drop_p)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # old checkpoints stored an explicit causal mask buffer
        state_dict.pop(prefix + 'mask', None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # causal scaled dot-product attention (B, N, T, D)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0, is_causal=True)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, deployment_prompt=False, no_prompt=False):
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.h_dim = h_dim

        ### transformer blocks
        input_seq_len = 1 + 3 * context_len
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.no_prompt = no_prompt
        if not no_prompt:
            self.embed_prompt = nn.Embedding(3, h_dim)
        self.deployment_prompt = deployment_prompt
        if deployment_prompt and not no_prompt:
            self.register_buffer('prompt_scale', torch.tensor([25.0, 1000.0]))
            self.embed_prompt_env = nn.Linear(2, h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        self.embed_state = torch.nn.Linear(state_dim, h_dim)

        # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        self.embed_action_1 = nn.Embedding(3, h_dim)
        self.embed_action_2 = nn.Embedding(4, h_dim)
        self.embed_action_3 = nn.Embedding(3, h_dim)
        use_action_tanh = False # False for discrete actions

        # # continuous actions
        # self.embed_action = torch.nn.Linear(act_dim, h_dim)
        # use_action_tanh = True # True for continuous actions

        ### prediction heads
        self.predict_rtg = torch.nn.Linear(h_dim, 1)
        self.predict_state = torch.nn.Linear(h_dim, state_dim)
        self.predict_action_1 = nn.Sequential(
            *([nn.Linear(h_dim, 3)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.predict_action_2 = nn.Sequential(
            *([nn.Linear(h_dim, 4)] + ([nn.Tanh()] if use_action_tanh else []))
        )
        self.predict_action_3 = nn.Sequential(
            *([nn.Linear(h_dim, 3)] + ([nn.Tanh()] if use_action_tanh else []))
        )


    def forward(self, timesteps, states, actions, returns_to_go, prompts):

        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings

        a1 = actions[..., 0].long()
        a2 = actions[..., 1].long()
        action_embeddings = self.embed_action_1(a1) + self.embed_action_2(a2)
        action_embeddings += time_embeddings
        # action_embeddings = self.embed_action_1(a1) + time_embeddings
        
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r_0, s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2 ...)
        h = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        # h = torch.stack(
        #     (state_embeddings, action_embeddings), dim=1
        # ).permute(0, 2, 1, 3).reshape(B, 2 * T, self.h_dim)

        # [B, 1, h_dim]
        if self.no_prompt:
            pass
        elif self.deployment_prompt:
            prompt_token = (self.embed_prompt(prompts[:, 0].long()) +
                            self.embed_prompt_env(prompts[:, 1:].float() / self.prompt_scale)).unsqueeze(1)
        else:
            prompt_token = self.embed_prompt(prompts.long()).unsqueeze(1)
        # [B, 1 + 3*T, h_dim]
        if not self.no_prompt:
            h = torch.cat((prompt_token, h), dim=1)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

        if not self.no_prompt:
            h = h[:, 1:, :]

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on the input sequence r_0, s_0, a_0 ... r_t, s_t, a_t
        # that is, for each timestep (t) we have 3 output embeddings from the transformer,
        # each conditioned on all previous timesteps plus 
        # the 3 input variables at that timestep (r_t, s_t, a_t) in sequence.
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)
        # h = h.reshape(B, T, 2, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        logits_1 = self.predict_action_1(h[:,1])
        logits_2 = self.predict_action_2(h[:,1])

        return [logits_1, logits_2]
