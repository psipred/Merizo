from torch import nn

from .nndef_ipa import InvariantPointAttention, StructureModuleTransition
from .nndef_ipa_primitives import LayerNorm, Linear

class ipa_block(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- IPA config ----
        self.c_s = 512
        self.c_z = 32
        self.c_ipa = 512
        self.no_qk_points = 4
        self.no_v_points = 8

        self.no_blocks = 6
        self.no_heads = 16
        self.no_transition_layers = 1
        self.dropout_rate = 0.0

        self.layer_norm_z = LayerNorm(self.c_z)
        self.layer_norm_s = LayerNorm(self.c_s)
        self.linear_in = Linear(self.c_s, self.c_s)

        self.ipa = InvariantPointAttention(
            c_s=self.c_s,
            c_z=self.c_z,
            c_hidden=self.c_ipa,
            no_heads=self.no_heads,
            no_qk_points=self.no_qk_points,
            no_v_points=self.no_v_points,
        )

        self.ipa_dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm_ipa = LayerNorm(self.c_s)

        self.transition = StructureModuleTransition(
            self.c_s,
            self.no_transition_layers,
            self.dropout_rate
        )

    def forward(
        self, s, z, rigids, mask=None):

        s = self.layer_norm_s(s)
        s = self.linear_in(s)
        
        z = self.layer_norm_z(z)
            
        mask = s.new_ones(s.shape[:-1]) if mask is None else mask

        for _ in range(self.no_blocks):
            s_, attn = self.ipa(s=s, z=z, r=rigids, mask=mask)
            
            s = s + s_  # [*, N, C_s]
            s = self.ipa_dropout(s)
            s = self.layer_norm_ipa(s)
            s = self.transition(s)

        return s, attn