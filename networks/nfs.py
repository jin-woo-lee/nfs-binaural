import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.functional as TAF
from einops import rearrange
from torch.nn.utils import weight_norm
import math

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import unfold
import torchaudio.transforms as TAT

import scipy.linalg
from scipy.spatial.transform import Rotation as R

class GeometricWarpfield(nn.Module):
    def __init__(self):
        super().__init__()

    def _transmitter_mouth(self, view):
        # offset between tracking markers and real mouth position in the dataset
        mouth_offset = np.array([0.09, 0, -0.20])
        quat = view[:, 3:, :].transpose(2, 1).contiguous().detach().cpu().view(-1, 4).numpy()
        # make sure zero-padded values are set to non-zero values (else scipy raises an exception)
        norms = scipy.linalg.norm(quat, axis=1)
        eps_val = (norms == 0).astype(np.float32)
        quat = quat + eps_val[:, None]
        transmitter_rot_mat = R.from_quat(quat)
        transmitter_mouth = transmitter_rot_mat.apply(mouth_offset, inverse=True)
        transmitter_mouth = torch.Tensor(transmitter_mouth).view(view.shape[0], -1, 3).transpose(2, 1).contiguous()
        if view.is_cuda:
            transmitter_mouth = transmitter_mouth.cuda()
        return transmitter_mouth

    def _3d_displacements(self, view):
        transmitter_mouth = self._transmitter_mouth(view)
        # offset between tracking markers and ears in the dataset
        left_ear_offset = torch.Tensor([0, -0.08, -0.22]).cuda() if view.is_cuda else torch.Tensor([0, -0.08, -0.22])
        right_ear_offset = torch.Tensor([0, 0.08, -0.22]).cuda() if view.is_cuda else torch.Tensor([0, 0.08, -0.22])
        # compute displacements between transmitter mouth and receiver left/right ear
        displacement_left = view[:, 0:3, :] + transmitter_mouth - left_ear_offset[None, :, None]
        displacement_right = view[:, 0:3, :] + transmitter_mouth - right_ear_offset[None, :, None]
        displacement = torch.stack([displacement_left, displacement_right], dim=1)
        return displacement

    def forward(self, view):
        ''' view: (batch, 7, frames)
            ->    (batch, 2, frames)
        '''
        distance = self._3d_displacements(view).pow(2).sum(2).pow(.5)
        delay = distance / 343.0 * 1000   # ms
        return delay

def get_inverse_window(forward_window, frame_length, frame_step):
    denom = torch.square(forward_window)
    overlaps = -(-frame_length // frame_step)  # Ceiling division.
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = denom.reshape(overlaps, frame_step)
    denom = denom.sum(0, keepdims=True)
    denom = denom.tile(overlaps, 1)
    denom = denom.reshape(overlaps * frame_step)
    return forward_window / denom[:frame_length]

class NFS(nn.Module):
    def __init__(self,
                 window_ms=200, nch=128, cdim=128, sr=48000,
                 wo_ni=False, wo_lff=False, wo_geowarp=False, wo_shifter=False, **kwargs):
        super().__init__()
        self.wo_ni      = wo_ni        # wo.NI
        self.wo_lff     = wo_lff       # wo.LFF
        self.wo_geowarp = wo_geowarp   # wo.GeoWarp
        self.wo_shifter = wo_shifter   # wo.Shifter
        taps = int(1e-3 * sr * window_ms)
        self.taps = taps
        self.cdim = cdim

        self.geom_ms = None if self.wo_geowarp else GeometricWarpfield()

        self.lff = None if self.wo_lff else LFF(7, self.cdim)
        self.enc = SinEncoding(self.cdim)
        max_ms = window_ms / 4
        if self.wo_lff:
            self.lfs = FourierShift(nch, taps, cdim, 1,max_ms, wo_shifter, wo_geowarp, wo_lff)
            self.rfs = FourierShift(nch, taps, cdim, 1,max_ms, wo_shifter, wo_geowarp, wo_lff)
        else:
            self.lfs = FourierShift(nch, taps, cdim,-1,max_ms, wo_shifter, wo_geowarp, wo_lff)
            self.rfs = FourierShift(nch, taps, cdim,-1,max_ms, wo_shifter, wo_geowarp, wo_lff)

        a_window = torch.hann_window(taps, periodic=True)
        s_window = get_inverse_window(a_window, taps, taps // 2)
        self.analysis_window  = nn.Parameter(a_window, requires_grad=False)
        self.synthesis_window = nn.Parameter(s_window, requires_grad=False)

        self.inject_noise = None if self.wo_ni else NoiseInjection()

    def forward(self, pos, dry):
        """ pos: (batch, 7, time)
            dry: (batch, 1, taps * frames)
        """
        bsz = pos.size(0)

        """ chunk into frames """
        dry = F.pad(dry, (self.taps // 2, self.taps // 2), mode='reflect')
        est = unfold(dry, self.taps // 2)      # (batch*frames, 1, taps)
        est = est * self.analysis_window.view(1,1,-1)
        n_chunks = est.size(0) // bsz
        n_frames = dry.size(-1) // (self.taps // 2)
        assert n_frames == n_chunks + 1, f"n_frames={n_frames}, n_chunks={n_chunks}"

        """ Geometric Warp """
        if not self.wo_geowarp:
            geo = self.geom_ms(pos)
            geo = F.interpolate(geo, size=n_frames-1)
            geo = rearrange(geo, 'b (c1 c2) t -> (b t) c1 c2', c1=2, c2=1)
            lgeo, rgeo = geo.chunk(2, 1)
        else:
            lgeo, rgeo = None, None

        """ Fourier Features """
        if not self.wo_lff:
            pos = rearrange(pos, 'b (p c) t -> (b t) p c', c=1)
            ff = self.lff(pos).transpose(2,1)      # (batch * time, 1, cdim)
            se = self.enc(pos)                     # (batch * time, 7, cdim)
            pos = torch.cat((ff,se), dim=1)        # (batch * time, 8, cdim)
        else:
            pos = rearrange(pos, 'b (p c) t -> (b t) p c', c=1)
            pos = self.enc(pos)                     # (batch * time, 7, cdim)

        """ Fourier Shift """
        l, lmag, lang = self.lfs(est, pos, lgeo, bsz)
        r, rmag, rang = self.rfs(est, pos, rgeo, bsz)
        est = torch.cat((l,r), dim=1)

        """ WOLA """
        est = rearrange(est, '(b t) c p -> b (c t) p', b=bsz)
        est = est * self.synthesis_window.view(1,1,-1)
        est = F.fold(est, (n_frames, self.taps // 2), (n_frames-1,1))
        est = est.narrow(2,1,est.size(2)-2)
        est = rearrange(est, 'b c t p -> b c (t p)')

        if not self.wo_ni:
            est = self.inject_noise(est)
        pos = rearrange(pos, '(b t) c p -> b c t p', b=bsz)
        return est, pos, [lmag, rmag], [lang, rang]

class SinEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        t_1 = torch.ones(1,1,dim).cumsum(-1).float().flip(-1)
        t_2 = torch.ones(1,1,dim).cumsum(-1).float()
        self.t_1 = nn.Parameter(t_1, requires_grad=False)
        self.t_2 = nn.Parameter(t_2, requires_grad=False)
        omega = torch.tensor([0.3, 0.3, 8.0, 8.0, 8.0, 0.3, 0.3]).float()
        self.omega = nn.Parameter(omega.view(1,-1,1), requires_grad=False)

    def forward(self, x):
        """ x: (batch, channel, 1)
            -> (batch, channel, dim)
        """
        return torch.sin(self.omega * self.t_1 * x) + torch.sin(2.0 * self.t_2 * x)

class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class LFF(nn.Module):
    def __init__(self, channel, lens):
        super().__init__()
        self.conv = nn.Conv1d(channel, lens, kernel_size=1, bias=False)
        self.sine = SinActivation()
        nn.init.uniform_(self.conv.weight, -np.sqrt(9 / channel), np.sqrt(9 / channel))

    def forward(self, x):
        x = self.conv(x)
        x = self.sine(x)
        return x

class Project(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        weights = torch.FloatTensor(out_ch, in_ch, 1).uniform_(- 1/in_ch, 1/in_ch)
        self.proj = nn.Parameter(weights, requires_grad=True)
        self.actv = nn.Softplus()

    def forward(self, x):
        x = F.conv1d(x, self.actv(self.proj))
        return x

class NoiseInjection(nn.Module):
    def __init__(self, nch=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, nch, 3, padding=1, bias=False),
            nn.Conv1d(nch, 1, 1, bias=False),
        )

    def forward(self, x):
        z = 1e-3 * self.conv(torch.randn_like(x))
        return x + z

class FourierShift(nn.Module):
    def __init__(self,
                 channel, taps, cdim, inf_ms, sup_ms,
                 wo_shifter=False, wo_geowarp=False, wo_lff=False, sr=48000):
        super().__init__()
        assert not(wo_geowarp and wo_shifter), \
        "Asserting both --wo_geowarp and --wo_shifter cannot be done"
        self.wo_shifter = wo_shifter
        self.wo_geowarp = wo_geowarp
        self.sr = sr
        self.taps = taps
        self.freq = taps // 2 + 1
        self.dims = self.freq

        self.channel = channel
        self.scale = ScaleEstimator(channel, cdim, self.freq, wo_lff)
        self.shift = ShiftEstimator(channel, cdim, self.freq, inf_ms, sup_ms, wo_lff) if not wo_shifter else None
        self.project = Project(channel, 1)

        omega = torch.linspace(0, - np.pi, self.freq).view(1,1,-1)
        self.omega = nn.Parameter(omega, requires_grad=False)

    def forward(self, x, c, g, b):
        """ x: (batch*frames, 1, taps)
            c: (batch*frames, 1, taps)
            g: (batch*frames, 1, 1)
            b: batch size (int)
            -> (batch*frames, 1, taps)
        """
        eps = torch.finfo(x.dtype).eps
        x = x.tile(1,self.channel,1)

        x = torch.fft.rfft(x, self.taps)     # (batch*frames, channel, freq)
        n = x.size(0) // b                   # frames

        if self.wo_shifter:
            ang = g                                         # (b*frames, ch, 1)
        elif self.wo_geowarp:
            ang = self.shift(c, b, n)                       # (b*frames, ch, freq)
        else:
            ang = g + self.shift(c, b, n)                   # (b*frames, ch, freq)
        mag = self.scale(c, b, n) / ang.clamp(min=1).pow(2) # (b*frames, ch, freq)
        omg = self.sr * ang * self.omega / 1000             # (b*frames, ch, freq)

        z = mag * torch.exp(1j * omg)
        x = torch.fft.irfft(x * z, self.taps)    # (batch*frames, channel, taps)

        x = self.project(x)
        return x, mag, ang

class ShiftEstimator(nn.Module):
    def __init__(self, channel, cdim, freq, inf_ms, sup_ms, wo_lff=False):
        super().__init__()
        self.freq = freq
        self.delta = inf_ms
        self.sigma = sup_ms
        nch = 3 if wo_lff else 4; d_ffn = cdim // 2
        self.mlp = nn.Sequential(
            AttnResBlock(nch, cdim, embed_dim=d_ffn),
            ConvSEBlock(nch, channel),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
        )

        self.scale = nn.Sigmoid()
        self.channel = channel

    def forward(self, x, b, n):
        """ x: (batch * frames, 7+1, width)
        """
        x = self.mlp(x)                   # b*t channel w
        x = self.scale(x).pow(2)       # b*t channel 1

        # upsample
        nch = x.size(1)
        x = rearrange(x, '(b t) c w -> (b c) w t', b=b)
        x = F.interpolate(x, size=n, mode='linear')
        x = rearrange(x, '(b c) w t -> (b t) c w', c=nch)

        if x.size(-1) > 1:
            x = F.interpolate(x, size=self.freq, mode='linear')
        x = x * (self.sigma-self.delta) + self.delta
        return x

class ScaleEstimator(nn.Module):
    def __init__(self, channel, cdim, freq, wo_lff):
        super().__init__()
        self.freq = freq
        nch = 3 if wo_lff else 4; d_ffn = cdim // 2
        self.mlp = nn.Sequential(
            AttnResBlock(nch, cdim, embed_dim=d_ffn),
            ConvSEBlock(nch, channel),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
            gMLPBlock(channel=channel, width=cdim, d_ffn=d_ffn),
        )

        self.scale = nn.Softplus()

    def forward(self, x, b, n):
        """ x: (batch * frames, 7+1, width)
        """
        x = self.mlp(x)                   # b*t channel w
        x = self.scale(x).pow(2)       # b*t channel 1

        # upsample
        nch = x.size(1)
        x = rearrange(x, '(b t) c w -> (b c) w t', b=b)
        x = F.interpolate(x, size=n, mode='linear')
        x = rearrange(x, '(b c) w t -> (b t) c w', c=nch)

        if x.size(-1) > 1:
            x = F.interpolate(x, size=self.freq, mode='linear')
        return x


class ConvSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.fc_1 = nn.Conv1d(out_ch, out_ch // 2, 1)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Conv1d(out_ch // 2, out_ch, 1)
        self.scale = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        c = x.mean(-1, keepdim=True)
        c = self.fc_1(c)
        c = self.relu(c)
        c = self.fc_2(c)
        c = self.scale(c)
        return c * x


class gMLPBlock(nn.Module):
    def __init__(self, channel, width, d_ffn=256):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc_1 = nn.Conv1d(channel, d_ffn, 1)
        self.gelu = nn.GELU()
        self.gate = SpatialGatingUnit(d_ffn, width)
        self.fc_2 = nn.Conv1d(d_ffn // 2, channel, 1)

    def forward(self, x):
        s = x
        x = self.norm(x)
        x = self.fc_1(x)
        x = self.gelu(x)
        x = self.gate(x)
        x = self.fc_2(x)
        x = x + s
        return x

class SpatialGatingUnit(nn.Module):
    def __init__(self, channel, dims):
        super().__init__()
        assert channel % 2 == 0, channel
        self.norm = nn.BatchNorm1d(channel // 2)
        self.proj = nn.Linear(dims, dims)
        torch.nn.init.constant_(self.proj.bias.data, 1)

    def forward(self, x):
        u, v = x.chunk(2, 1)
        v = self.norm(v)
        v = self.proj(v)
        return u * v


class AttnResBlock(nn.Module):
    def __init__(self, channel, input_dim, embed_dim):
        super().__init__()
        assert channel < 7
        self.ch_1 = channel
        self.ch_2 = 7 - channel

        self.fc_1 = nn.Conv1d(self.ch_1, 16, 1, bias=False)
        self.fc_2 = nn.Conv1d(self.ch_2, 16, 1, bias=False)
        self.attn = CrossAttention(input_dim, input_dim, embed_dim)
        self.fc_3 = nn.Conv1d(16, self.ch_1, 1, bias=False)

    def forward(self, x):
        p = x.narrow(1,0,self.ch_1)    # (lff, x, y, z)
        c = x.narrow(1,3,self.ch_2)    # (qx, qy, qz, qw)

        q = self.fc_1(p)
        c = self.fc_2(c)
        x = self.attn(q, c)
        x = self.fc_3(x)
        return p + x

class CrossAttention(nn.Module):
    def __init__(self, qry_dim, con_dim, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(qry_dim, embed_dim)
        self.k_proj = nn.Linear(con_dim, embed_dim)
        self.v_proj = nn.Linear(con_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, qry_dim)
        
        self.embed_dim = embed_dim
        
    def forward(self, x, c):
    
        query, key, value = x, c, c
        bsz, nch, qry_dim = query.size()
        
        q = self.q_proj(query)    # (B, C, embed_dim)
        k = self.k_proj(key)      # (B, C, embed_dim)
        v = self.v_proj(value)    # (B, C, embed_dim)
        
        q = rearrange(q, 'b c w -> b w c')
        attn_weights = torch.bmm(q, k) * (nch ** (-0.5))
        assert list(attn_weights.size()) == [bsz, self.embed_dim, self.embed_dim]

        attn_weights = torch.softmax(attn_weights, dim=-1)

        v = rearrange(v, 'b c w -> b w c')
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz, self.embed_dim, nch]

        attn = rearrange(attn, 'b w c -> b c w')
        attn = self.out_proj(attn)

        return attn



def test():
    #m = GeometricWarpfield()
    #a = torch.randn(2,7,10)
    #b = m(a)

    m = NFS()
    a = torch.randn(1,7,int(0.8*120))
    b = torch.randn(1,1,int(0.8*48000))
    c = m(a, b)

if __name__ == '__main__':
    test()

