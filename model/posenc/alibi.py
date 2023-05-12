import torch
from math import log2, floor
from torch import nn
from einops import rearrange

# Symmetric ALiBi relative positional bias
class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, slope_factor):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)
        
        self.slope_factor = slope_factor

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** floor(log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, residx, clip=False):
        relative_position = residx.unsqueeze(0) - residx.unsqueeze(1)
        bias = torch.abs(relative_position)

        # Clip attention bias offsets at 32, along lines of AlphaFold2 relative pos encoding
        bias = bias.clip(max=32) if clip else bias
        bias = bias.unsqueeze(0).expand(1, self.heads, -1, -1)

        return bias * -(self.slopes * self.slope_factor)