import os
import argparse
import glob
from einops import rearrange
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset.loader import load_txt
from utils import unfold, filter_dict
from networks.nfs import get_inverse_window

def unfold_batch(x, window, n_ch=1):
    taps = window.size(-1)
    x = F.pad(x, (taps // 2, taps // 2), mode='reflect')
    x = unfold(x, taps // 2, n_ch=n_ch)    # (batch*frames, 1, taps)
    return x * window

def fold_batch(x, window):
    taps = window.size(-1)
    n_frames = x.size(0) + 1
    x = rearrange(x, 't c p -> c t p')
    x = x * window
    x = F.fold(x, (n_frames, taps // 2), (n_frames-1,1))
    x = x.narrow(2,1,x.size(2)-2)
    x = rearrange(x, 'c b t p -> c (b t p)')
    return x.transpose(0,1)

def pad_to_lens(x, p, lens_sec):
    x = torch.from_numpy(x).view(1,1,-1)
    p = p.transpose(0,1).unsqueeze(0)
    x_lens = int(lens_sec * 48000)
    p_lens = int(lens_sec * 120)
    x_res = x_lens - x.size(-1) % x_lens
    p_res = p_lens - p.size(-1) % p_lens
    x = F.pad(x, (0, x_res))
    p = F.pad(p, (0, p_res))
    return x.float().cuda(), p.float().cuda()


def get_quaternion_like(X, degree):
    Qx = Qy = torch.zeros_like(X)
    Qw = + torch.cos(degree / 2) * torch.ones_like(X)
    Qz = - torch.sin(degree / 2) * torch.ones_like(X)
    return Qw, Qx, Qy, Qz

def circular_position(time, vel, radius=1, offset=0, sr=120):
    t = torch.arange(time) / sr
    theta = 2 * np.pi * vel * t + offset
    x = radius * torch.cos(theta).view(1,-1)
    y = radius * torch.sin(theta).view(1,-1)
    z = -0.05 * torch.ones_like(x)

    d = np.pi + theta.unsqueeze(0)
    Qw, Qx, Qy, Qz = get_quaternion_like(x, d)
    return torch.cat((x, y, z, Qw, Qx, Qy, Qz), dim=0)   # (7, time)

def load_audio(path):
    x, sr = sf.read(path)
    if len(list(x.shape)) > 1:
        d = np.argmin(list(x.shape))
        x = np.sum(x, axis=d) / x.shape[d]
    x = torch.from_numpy(x).view(1,1,-1).float().cuda()
    if sr != 48000:
        x = TAF.resample(x, sr, 48000, resampling_method='kaiser_window')
    res = x.size(-1) % 48000
    if res > 0:
        x = F.pad(x, (0,48000-res))
    return x

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--name', type=str, default="nfs")
    parser.add_argument('--lens_sec', type=float, default=1.0)
    parser.add_argument('--lens', type=float, default=None)
    parser.add_argument('--model_window_ms', type=float, default=200)
    parser.add_argument('--channel', type=int, default=128)
    parser.add_argument('--cdim', type=int, default=128)

    parser.add_argument('--root_dir', type=str, default=None, help="directory that contains mono wav files and paired position txt files")
    parser.add_argument('--save_dir', type=str, default=None, help="directory to save results")

    return parser

def inference(args, nfs):

    args.lens = int(args.lens_sec * 48000)
    pos_lens = int(args.lens_sec * 120)
    taps = args.lens
    a_window = torch.hann_window(taps, periodic=True).cuda().view(1,1,-1)
    s_window = get_inverse_window(a_window, taps, taps // 2).cuda().view(1,1,-1)

    root_dir = args.root_dir
    save_dir = args.save_dir
    paths = sorted(glob.glob(f"{root_dir}/*.wav"))

    nfs.eval()
    iterator = tqdm(paths)
    for i, dp in enumerate(iterator):
        x = load_audio(dp)
        fname = dp.split('/')[-1].split('.')[0]
        subset = dp.split('/')[-2]
        iterator.set_description(f"Rendering {subset}")
        try:
            p = load_txt(f"{root_dir}/{subset}/{fname}.txt")  # (time, channel)
            p = p.transpose(0,1).unsqueeze(0).float().cuda()  # (1, channel, time)
        except FileNotFoundError:
            time = int(120 * (x.size(2) / 48000))
            sec_per_rotation = 10
            radius = 1.0
            offset = np.pi * i / (len(iterator)-1)
            p = circular_position(time, 1/sec_per_rotation, radius, offset)
            p = p.unsqueeze(0).float().cuda() # (1, channel, time)

        p_taps = int(taps / 48000 * 120)
        z = unfold_batch(x, a_window)
        p = unfold_batch(p, torch.ones_like(a_window.narrow(-1,0,p_taps)), n_ch=7)
        o = []
        dur = []
        for b in range(z.size(0)):
            with torch.no_grad():
                o.append(nfs(p.narrow(0,b,1), z.narrow(0,b,1))[0])
        y = torch.cat(o, dim=0)
        y = fold_batch(y, s_window)
        y = y.cpu().numpy()

        data_dir = os.path.join(save_dir, subset)
        os.makedirs(data_dir, exist_ok=True)
        sf.write(f'{data_dir}/{fname}.wav', y, samplerate=48000, subtype="PCM_16")
        sf.write(f'{data_dir}/{fname}-mono.wav', x.cpu().view(-1).numpy(), samplerate=48000, subtype="PCM_16")

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    gen = __import__(f'networks.nfs', fromlist=[''])

    nfs = gen.NFS(window_ms=args.model_window_ms, nch=args.channel, cdim=args.cdim)
    n_params = sum([param.view(-1).size()[0] for param in nfs.parameters()])
    nfs.load_state_dict(filter_dict(torch.load(args.ckpt)["nfs"]))
    nfs = nfs.to('cuda:0')
    inference(args, nfs)

