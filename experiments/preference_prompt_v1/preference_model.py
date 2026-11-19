"""Existing DT backbone plus optional continuous scalar preference prefix."""
import torch
from torch import nn
from decision_transformer.model import DecisionTransformer

class PreferenceTransformer(DecisionTransformer):
    def __init__(self, enabled):
        super().__init__(110, 2, 3, 128, 8, 1, .1, max_timestep=10080, no_prompt=True)
        self.preference_enabled = enabled
        if enabled:
            self.preference_projection = nn.Linear(1, self.h_dim)

    def forward(self, timesteps, states, actions, returns_to_go, preference):
        batch, length, _ = states.shape
        time = self.embed_timestep(timesteps)
        state = self.embed_state(states) + time
        action = self.embed_action_1(actions[...,0].long()) + self.embed_action_2(actions[...,1].long()) + time
        rtg = self.embed_rtg(returns_to_go) + time
        tokens = torch.stack((rtg, state, action), dim=2).reshape(batch, 3*length, self.h_dim)
        if self.preference_enabled:
            prefix = self.preference_projection(preference.float().reshape(batch,1) / 2.5).unsqueeze(1)
            tokens = torch.cat((prefix,tokens), dim=1)
        hidden = self.transformer(self.embed_ln(tokens))
        if self.preference_enabled:
            hidden = hidden[:,1:]
        hidden = hidden.reshape(batch,length,3,self.h_dim)[:,:,1]
        return [self.predict_action_1(hidden), self.predict_action_2(hidden)]
