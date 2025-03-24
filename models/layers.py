import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, c, _, _ = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, c, 1, 1) * y + bias.view(1, c, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        _, c, _, _ = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, c, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None
    
    
class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
    
class SCIUnit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(nn.Conv2d(in_features, in_features // 5, 1, 1, 0),
                        nn.GELU(),
                        nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),
                        nn.GELU(),
                        nn.Conv2d(in_features // 5, out_features, 1, 1, 0))
        
        self.linear = nn.Conv2d(in_features, out_features,1,1,0)

    def forward(self, x):
        x = self.conv(x) * self.linear(x)
        return x


class ImprovedNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg1 = SCIUnit(dw_channel,dw_channel//2)
        
        ffn_channel = FFN_Expand * c

        self.sg2 = SCIUnit(ffn_channel,ffn_channel//2)

        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
    

class WaveletNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.wavelet_block1 = WKLModule(c, wavelet='haar')

        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.wavelet_block1(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        # gate
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MultiInputMultiScaleConv(nn.Module):
    def __init__(self, in_channels, s):
        super(MultiInputMultiScaleConv, self).__init__()
    
        self.s = s
        # (s, 2s)
        self.conv_a = nn.Conv2d(in_channels, in_channels, kernel_size=(s, 2 * s), padding=0, bias=False)
        # (2s, s)
        self.conv_b = nn.Conv2d(in_channels, in_channels, kernel_size=(2 * s, s), padding=0, bias=False)
        # (2s, 2s)
        self.conv_c = nn.Conv2d(in_channels, in_channels, kernel_size=(2 * s, 2 * s), padding=0, bias=False)
    
    def calculate_padding(self, kernel_size):
        pad = kernel_size - 1
        padding_left = pad // 2
        padding_right = pad - padding_left
        return padding_left, padding_right
    
    def forward(self, a, b, c):
        pad_h_a = self.calculate_padding(self.s)
        pad_w_a = self.calculate_padding(2 * self.s)
        padding_a = (pad_w_a[0], pad_w_a[1], pad_h_a[0], pad_h_a[1]) 
        a_padded = F.pad(a, padding_a)
        out_a = self.conv_a(a_padded)
        
        pad_h_b = self.calculate_padding(2 * self.s)
        pad_w_b = self.calculate_padding(self.s)
        padding_b = (pad_w_b[0], pad_w_b[1], pad_h_b[0], pad_h_b[1])
        b_padded = F.pad(b, padding_b)
        out_b = self.conv_b(b_padded)
        
        pad_h_c = self.calculate_padding(2 * self.s)
        pad_w_c = self.calculate_padding(2 * self.s)
        padding_c = (pad_w_c[0], pad_w_c[1], pad_h_c[0], pad_h_c[1])
        c_padded = F.pad(c, padding_c)
        out_c = self.conv_c(c_padded)
        
        out = torch.cat([out_a, out_b, out_c], dim=1)
        return out


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1):
        super(DWT, self).__init__()
        self.wavelet = pywt.Wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi
        self.level = level

    def forward(self, x):
        _, c, _, _ = x.shape
        wavelet_components = []

        ll = x
        dwt_kernel = make_2dfilters(lo=self.dec_lo, hi=self.dec_hi).repeat(c, 1, 1).unsqueeze(dim=1)

        for _ in range(self.level):
            ll = adaptive_pad2d(ll, self.wavelet)
            high_component = F.conv2d(ll, dwt_kernel, stride=2, groups=c)
            reshaped_components = rearrange(high_component, 'b (c f) h w -> b c f h w', f=4)
            ll, lh, hl, hh = reshaped_components.split(1, 2)
            wavelet_components.append((lh.squeeze(2), hl.squeeze(2), hh.squeeze(2)))

        wavelet_components.append(ll.squeeze(2))
        return wavelet_components[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        ll = x[0]
        _, c, _, _ = ll.shape
        idwt_kernel = make_2dfilters(lo=self.rec_lo, hi=self.rec_hi).repeat(c, 1, 1).unsqueeze(dim=1)
        self.filter_size = idwt_kernel.shape[-1]
        
        for idx, component_h in enumerate(x[1:]):
            ll = rearrange(torch.cat([ll.unsqueeze(2), *[component.unsqueeze(2) for component in component_h]], 2), 'b c f h w -> b (c f) h w')
            ll = F.conv_transpose2d(ll, idwt_kernel, stride=2, groups=c)

            pad_left = pad_right = pad_top = pad_bottom = (2 * self.filter_size - 3) // 2

            if idx < len(x) - 2:
                pred_len, next_len = ll.shape[-1] - (pad_left + pad_right), x[idx + 2][0].shape[-1]
                pred_len2, next_len2 = ll.shape[-2] - (pad_top + pad_bottom), x[idx + 2][0].shape[-2]

                if next_len != pred_len:
                    pad_right += 1
                if next_len2 != pred_len2:
                    pad_bottom += 1

            ll = ll[..., pad_top: -pad_bottom if pad_bottom > 0 else None, 
               pad_left: -pad_right if pad_right > 0 else None]

        return ll


class WKLModule(nn.Module):
    def __init__(self, dim, wavelet='haar'):
        super(WKLModule, self).__init__()
        self.dim = dim
        self.wavelet = pywt.Wavelet(wavelet)
        
        dec_lo, dec_hi, rec_lo, rec_hi = get_wavelet_filters(wavelet)
        self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
        self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
        self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
        self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*3, dim*6, 1)
        self.conv2 = nn.Conv2d(dim*6, dim*6, 7, padding=3, groups=dim*6)
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*6, dim*4, 1)

        self.conv_hvd = MultiInputMultiScaleConv(dim,s=2)
        self.conv_ya = nn.Conv2d(dim, 3*dim,kernel_size=1, bias=False)

    def forward(self, x):
        ya, (yh, yv, yd) = self.wavedec(x)
        y_hvd = self.conv_hvd(yh, yv, yd)
        
        dec_x = self.conv_ya(ya) + y_hvd
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)])
        return y


def get_wavelet_filters(wavelet):
    wavelet = pywt.Wavelet(wavelet)
    dtype = torch.float32
    device = 'cpu'
    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank
    dec_lo_tensor = torch.tensor(dec_lo[::-1], device=device, dtype=dtype).unsqueeze(0)
    dec_hi_tensor = torch.tensor(dec_hi[::-1], device=device, dtype=dtype).unsqueeze(0)
    rec_lo_tensor = torch.tensor(rec_lo[::-1], device=device, dtype=dtype).unsqueeze(0)
    rec_hi_tensor = torch.tensor(rec_hi[::-1], device=device, dtype=dtype).unsqueeze(0)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def make_2dfilters(lo, hi):
    ll = torch.unsqueeze(torch.reshape(lo, [-1]), dim=-1) * torch.unsqueeze(torch.reshape(lo, [-1]), dim=0)
    lh = torch.unsqueeze(torch.reshape(hi, [-1]), dim=-1) * torch.unsqueeze(torch.reshape(lo, [-1]), dim=0)
    hl = torch.unsqueeze(torch.reshape(lo, [-1]), dim=-1) * torch.unsqueeze(torch.reshape(hi, [-1]), dim=0)
    hh = torch.unsqueeze(torch.reshape(hi, [-1]), dim=-1) * torch.unsqueeze(torch.reshape(hi, [-1]), dim=0)

    filter2d = torch.stack([ll, lh, hl, hh], 0)
    return filter2d


def adaptive_pad2d(input_data, wavelet):
    filter_size = len(wavelet.dec_lo)

    pad_left = pad_right = pad_top = pad_bottom = (2 * filter_size - 3) // 2

    if input_data.shape[-2] % 2 != 0:
        pad_bottom += 1
    if input_data.shape[-1] % 2 != 0:
        pad_right += 1
    padded_data = F.pad(input_data, [pad_left, pad_right, pad_top, pad_bottom], mode='replicate')
    
    return padded_data


