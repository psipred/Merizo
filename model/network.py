from torch import nn

from .ipa.ipa_encoder import ipa_block
from .ipa.nndef_ipa_primitives import Rigid, Rotation

from .posenc.alibi import AlibiPositionalBias
from .decoders.mask_decoder import MaskTransformer


class Merizo(nn.Module):
    def __init__(self):
        super(Merizo, self).__init__()

        self.no_classes = 20

        self.linear_s_in = nn.Linear(20, 512, bias=False)
        self.linear_z_in = nn.Linear(1, 32, bias=False)

        self.ipa = ipa_block()
        self.alibi = AlibiPositionalBias(heads=16, slope_factor=1)

        self.decoder_head = MaskTransformer(
            n_cls=self.no_classes,
            n_layers=10,
            n_heads=16,
            d_model=512,
            d_ff=512,
            drop_path_rate=0,
            dropout=0,
        )

    def forward(self, features, mask=None):

        s, z, r, t, ri = features['s'], features['z'], features['r'], features['t'], features['ri']

        if mask is not None:
            s = s[:, mask]
            z = z[:, mask][:, :, mask]
            r = r[:, mask]
            t = t[:, mask]
            ri = ri[:, mask]

        ipa_out, _ = self.ipa(
            s=self.linear_s_in(s),
            z=self.linear_z_in(z),
            rigids=Rigid(Rotation(r), t),
        )

        domain_ids, conf_res = self.decoder_head(
            ipa_out,
            bias=self.alibi(ri.squeeze(0), clip=True)
        )

        return domain_ids, conf_res
