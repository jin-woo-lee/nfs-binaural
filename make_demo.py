import os
import argparse
import glob
from einops import rearrange
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import librosa.display
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset.loader import load_txt
from utils import unfold, filter_dict
from networks.nfs import get_inverse_window
import shutil
import subprocess
import matplotlib
matplotlib.use("agg")

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
    parser.add_argument('--is_eval_set', action='store_true', help='whether to inference over evaluation dataset (filename convension is a little bit different)')

    return parser

def inference(args, nfs):

    args.lens = int(args.lens_sec * 48000)
    pos_lens = int(args.lens_sec * 120)
    taps = args.lens
    a_window = torch.hann_window(taps, periodic=True).cuda().view(1,1,-1)
    s_window = get_inverse_window(a_window, taps, taps // 2).cuda().view(1,1,-1)

    if args.is_eval_set:
        root_dir = args.root_dir
        save_dir = args.save_dir
        paths = sorted(glob.glob(f"{root_dir}/*/mono.wav"))
    else:
        root_dir = args.root_dir
        save_dir = args.save_dir
        paths = sorted(glob.glob(f"{root_dir}/*/*.wav"))

    nfs.eval()
    iterator = tqdm(paths)
    for i, dp in enumerate(iterator):
        x, _ = sf.read(dp)
        fname = dp.split('/')[-1].split('.')[0]
        subset = dp.split('/')[-2]
        iterator.set_description(f"Rendering {subset}")
        #------------------------------ 
        if not args.is_eval_set:
            p = load_txt(f"{root_dir}/{subset}/{fname}.txt")  # (time, channel)
            x = torch.from_numpy(x).view(1,1,-1).float().cuda()
            p = p.transpose(0,1).unsqueeze(0).float().cuda()       # (1, channel, time)
        else:
            p = load_txt(f"{root_dir}/{subset}/tx_positions.txt")  # (time, channel)
            x, p = pad_to_lens(x, p, args.lens_sec)
            fname = 'binauralized'

        p_taps = int(taps / 48000 * 120)
        z = unfold_batch(x, a_window)
        p = unfold_batch(p, torch.ones_like(a_window.narrow(-1,0,p_taps)), n_ch=7)
        o = []; lir, rir, lmg, rmg = [], [], [], []
        dur = []
        sr = nfs.lfs.sr; omega = nfs.lfs.omega
        for b in range(z.size(0)):
            with torch.no_grad():
                out, _, lm, la = nfs(p.narrow(0,b,1), z.narrow(0,b,1))
                lmag, rmag = lm
                lang, rang = la
                lomg = sr * lang * omega / 1000
                romg = sr * rang * omega / 1000

                lz = torch.fft.irfft((lmag * torch.exp(1j * lomg)).sum(1), nfs.lfs.taps)
                rz = torch.fft.irfft((rmag * torch.exp(1j * romg)).sum(1), nfs.rfs.taps)
                lz = lz.narrow(0,0,lz.size(0)-1)
                rz = rz.narrow(0,0,rz.size(0)-1)

                o.append(out)
                lir.append(lz.cpu()); rir.append(rz.cpu())
        y = torch.cat(o, dim=0)
        y = fold_batch(y, s_window).cpu().numpy()

        lir = torch.cat(lir, dim=0)
        rir = torch.cat(rir, dim=0)
        lmg = 20 * (torch.fft.rfft(lir).abs() + 1e-12).log10().numpy()
        rmg = 20 * (torch.fft.rfft(rir).abs() + 1e-12).log10().numpy()
        ldy = (torch.argmax(lir.abs(), dim=-1) / sr * 1000).numpy()
        rdy = (torch.argmax(rir.abs(), dim=-1) / sr * 1000).numpy()
        lir = lir.numpy()
        rir = rir.numpy()

        fax = np.linspace(0, sr // 2, lmg.shape[-1]) / 1000

        data_dir = os.path.join(save_dir, subset)
        os.makedirs(data_dir, exist_ok=True)
        sf.write(f'{data_dir}/{fname}.wav', y, samplerate=48000, subtype="PCM_16")

        freq_ticks = [0,2,4,8,16]
        for j in range(lir.shape[0]):
            f, ax = plt.subplots(figsize=(5,7), nrows=3, ncols=2)
            ax[0,0].set_title("Left Ear"); ax[0,1].set_title("Right Ear")

            # Impulse Response
            ax[0,0].set_ylabel("Impulse Response")
            ax[0,0].plot(lir[j], c='k', lw=0.5)
            ax[0,1].plot(rir[j], c='k', lw=0.5)
            ax[0,0].axhline(y=0, c='k', lw=0.3)
            ax[0,1].axhline(y=0, c='k', lw=0.3)
            ax[0,0].set_ylim((-1,1))
            ax[0,1].set_ylim((-1,1))
            ax[0,0].set_xticks([])
            ax[0,1].set_xticks([])
            ax[0,0].set_yticks([])
            ax[0,1].set_yticks([])

            # Magnitude Response
            ax[1,0].set_ylabel("Magnitude Response (dB)")
            ax[1,0].plot(fax, lmg[j], c='k', lw=0.5)
            ax[1,1].plot(fax, rmg[j], c='k', lw=0.5)
            ax[1,0].axhline(y=0, c='k', lw=0.3)
            ax[1,1].axhline(y=0, c='k', lw=0.3)
            for ft in freq_ticks:
                ax[1,0].axvline(x=ft, c='k', lw=0.3)
                ax[1,1].axvline(x=ft, c='k', lw=0.3)
            ax[1,0].set_ylim((-60,30))
            ax[1,1].set_ylim((-60,30))
            ax[1,0].set_xticks([])
            ax[1,1].set_xticks([])
            ax[1,0].set_yticks([])
            ax[1,1].set_yticks([])

            # Delay
            ax[2,0].set_ylabel("Delay (ms)")
            lbar = ax[2,0].barh(np.array([0]), ldy[j])
            rbar = ax[2,1].barh(np.array([0]), rdy[j])
            ax[2,0].axvline(rdy[j], c='k', lw=0.3)
            ax[2,1].axvline(ldy[j], c='k', lw=0.3)
            ax[2,0].set_xlim((0,5))
            ax[2,1].set_xlim((0,5))
            ax[2,0].bar_label(lbar)
            ax[2,1].bar_label(rbar)
            ax[2,0].set_xticks([])
            ax[2,1].set_xticks([])
            ax[2,0].set_yticks([])
            ax[2,1].set_yticks([])


            plt.tight_layout()
            plt.subplots_adjust(wspace=0.)
            plt.subplots_adjust(hspace=0.)

            os.makedirs(f'{data_dir}/temp', exist_ok=True)
            plt.savefig(data_dir + '/temp/file%02d.png' % j)
            plt.clf()
            plt.close("all")

        subprocess.call([
            'ffmpeg', '-framerate', '20',
            '-i', f'{data_dir}/temp/file%02d.png',
            '-r', '30', '-pix_fmt', 'yuv420p',
            f'{data_dir}/silent_video.mp4'
        ])
        subprocess.call([
            'ffmpeg',
            '-i', f'{data_dir}/silent_video.mp4',
            '-i', f'{data_dir}/{fname}.wav',
            '-c:v', 'copy', '-map', '0:v', '-map', '1:a',
            '-shortest', '-y',
            f'{data_dir}/{fname}.mp4'
        ])

        shutil.rmtree(f"{data_dir}/temp")

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = f'{args.gpu}'

    gen = __import__(f'networks.nfs', fromlist=[''])

    nfs = gen.NFS(window_ms=args.model_window_ms, nch=args.channel, cdim=args.cdim)
    n_params = sum([param.view(-1).size()[0] for param in nfs.parameters()])
    print(f"num. params: {n_params}")
    nfs.load_state_dict(filter_dict(torch.load(args.ckpt)["nfs"]))
    nfs = nfs.to('cuda:0')
    inference(args, nfs)

