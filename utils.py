import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pickle
import copy
import scipy
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TAF
import collections
import soundfile as sf
from torch.autograd import grad

from sklearn import metrics
from optimizer import AdamW, lr_policy, Novograd
from einops import rearrange

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def save_samples(sample, directory, name, sr, b=0):
    dry_mag, wet_mag, est_mag = sample[0]
    dry_mel, wet_mel, est_mel = sample[1]
    dry_wav, wet_wav, est_wav = sample[2]
    wet_ild, est_ild, pos_enc = sample[3]

    pe1 = rearrange(pos_enc[0,:4], 'c t p -> t (c p)').T
    pe2 = rearrange(pos_enc[0,4:], 'c t p -> t (c p)').T

    fig1, ax = plt.subplots(6, 3, figsize=(7,20))
    #---------- 
    dl  = librosa.display.specshow(dry_mag[b,0], cmap='magma', ax=ax[0,0])
    wgl = librosa.display.specshow(wet_mag[b,0], cmap='magma', ax=ax[1,0])
    wgr = librosa.display.specshow(wet_mag[b,1], cmap='magma', ax=ax[2,0])
    wml = librosa.display.specshow(wet_mel[b,0], cmap='magma', ax=ax[3,0])
    wmr = librosa.display.specshow(wet_mel[b,1], cmap='magma', ax=ax[4,0])
    #---------- 
    pe1 = librosa.display.specshow(pe1,          cmap='bwr',   ax=ax[0,1])
    egl = librosa.display.specshow(est_mag[b,0], cmap='magma', ax=ax[1,1])
    egr = librosa.display.specshow(est_mag[b,1], cmap='magma', ax=ax[2,1])
    eml = librosa.display.specshow(est_mel[b,0], cmap='magma', ax=ax[3,1])
    emr = librosa.display.specshow(est_mel[b,1], cmap='magma', ax=ax[4,1])
    #---------- 
    dif_dry = dry_mag[b,0] - wet_mag[b,0]
    dif_mag = wet_mag - est_mag; dif_mel = wet_mel - est_mel
    pe2 = librosa.display.specshow(pe2,          cmap='bwr', ax=ax[0,2])
    dgl = librosa.display.specshow(dif_mag[b,0], cmap='bwr', ax=ax[1,2])
    dgr = librosa.display.specshow(dif_mag[b,1], cmap='bwr', ax=ax[2,2])
    dml = librosa.display.specshow(dif_mel[b,0], cmap='bwr', ax=ax[3,2])
    dmr = librosa.display.specshow(dif_mel[b,1], cmap='bwr', ax=ax[4,2])
    #---------- 
    dif_ild = wet_ild - est_ild
    ax[5,0].plot(wet_ild[b], 'g-'); ax[5,0].axhline(y=0, color='k', lw=.3)
    ax[5,1].plot(est_ild[b], 'k-'); ax[5,1].axhline(y=0, color='k', lw=.3)
    ax[5,1].plot(wet_ild[b], 'g:')
    ax[5,2].plot(dif_ild[b], 'k-'); ax[5,2].axhline(y=0, color='k', lw=.3)
    #---------- 
    ax[0,0].set_title("dry mag");      ax[0,1].set_title("pos enc")
    ax[1,0].set_title("wet mag L");    ax[1,1].set_title("est mag L")
    ax[2,0].set_title("wet mag R");    ax[2,1].set_title("est mag R")
    ax[3,0].set_title("wet mel L");    ax[3,1].set_title("est mel L")
    ax[4,0].set_title("wet mel R");    ax[4,1].set_title("est mel R")
    ax[0,2].set_title("pos enc")
    ax[1,2].set_title("diff mag L")
    ax[2,2].set_title("diff mag R")
    ax[3,2].set_title("diff mel L")
    ax[4,2].set_title("diff mel R")
    plt.colorbar(dl,  ax=ax[0,0]); plt.colorbar(pe1, ax=ax[0,1])
    plt.colorbar(wgl, ax=ax[1,0]); plt.colorbar(egl, ax=ax[1,1])
    plt.colorbar(wgr, ax=ax[2,0]); plt.colorbar(egr, ax=ax[2,1])
    plt.colorbar(wml, ax=ax[3,0]); plt.colorbar(eml, ax=ax[3,1])
    plt.colorbar(wmr, ax=ax[4,0]); plt.colorbar(emr, ax=ax[4,1])
    plt.colorbar(pe2, ax=ax[0,2])
    plt.colorbar(dgl, ax=ax[1,2])
    plt.colorbar(dgr, ax=ax[2,2])
    plt.colorbar(dml, ax=ax[3,2])
    plt.colorbar(dmr, ax=ax[4,2])
    min_ild = min(wet_ild[b].min(), est_ild[b].min()) - 1e-2
    max_ild = max(wet_ild[b].max(), est_ild[b].max()) + 1e-2
    glb_max = max(abs(min_ild), max_ild)
    ax[5,0].set_ylim(min_ild, max_ild)
    ax[5,1].set_ylim(min_ild, max_ild)
    ax[5,2].set_ylim(- glb_max, glb_max)
    dmin = min(dry_mag[b,0].min(),dry_mel[b,0].min())
    dmax = max(dry_mag[b,0].max(),dry_mel[b,0].max())
    wgmin, wgmax = wet_mag[b].min(), wet_mag[b].max()
    wmmin, wmmax = wet_mel[b].min(), wet_mel[b].max()
    dl.set_clim(dmin, dmax)
    wgl.set_clim(wgmin, wgmax); egl.set_clim(wgmin, wgmax)
    wgr.set_clim(wgmin, wgmax); egr.set_clim(wgmin, wgmax)
    wml.set_clim(wmmin, wmmax); eml.set_clim(wmmin, wmmax)
    wmr.set_clim(wmmin, wmmax); emr.set_clim(wmmin, wmmax)
    dgl.set_clim(-40, 40); dgr.set_clim(-40, 40)
    dml.set_clim(-40, 40); dmr.set_clim(-40, 40)
    pe1.set_clim(-2, 2); pe2.set_clim(-2, 2)
    #---------- 
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{name}-{b}-gen.png")
    plt.close()

    figs = [fig1]

    dry_wav = np.transpose(dry_wav[b])
    wet_wav = np.transpose(wet_wav[b])
    est_wav = np.transpose(est_wav[b])
    sf.write(f"{directory}/{name}-{b}-dry.wav", dry_wav, samplerate=sr,subtype='PCM_16')
    sf.write(f"{directory}/{name}-{b}-wet.wav", wet_wav, samplerate=sr,subtype='PCM_16')
    sf.write(f"{directory}/{name}-{b}-est.wav", est_wav, samplerate=sr,subtype='PCM_16')

    wavs = [dry_wav, wet_wav, est_wav]

    return figs, wavs


def filter_dict(dicts):
    filtered_dicts = {}
    for key in dicts.keys():
        new_key = key.split('module.')[-1] if 'module' in key else key
        filtered_dicts[new_key] = dicts[key]
    return filtered_dicts

def get_optimizer(optimizer_name, model_parameters, config):

    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_parameters, **config)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_parameters, **config)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model_parameters, **config)
    elif optimizer_name == "radam":
        optimizer = torch.optim.RAdam(model_parameters, **config)
    elif optimizer_name == "novograd":
        optimizer = Novograd(model_parameters, **config)
    else:
        print('Unknown optimizer', optimizer_name)
        sys.exit()
    return optimizer

def get_scheduler(scheduler_name, optimizer, config):
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **config)
    elif scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **config)
    elif scheduler_name == 'sgdr':
        scheduler = SGDRScheduler(optimizer, **config)
    elif scheduler_name == 'lambda_lr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, **config)
    elif scheduler_name == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config)
    elif scheduler_name == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **config)
    else:
        scheduler = None
    return scheduler

def cal_rms(amp, eps=1e-5):
    if isinstance(amp, torch.Tensor):
        return amp.pow(2).mean(-1, keepdim=True).pow(.5)
    elif isinstance(amp, np.ndarray):
        return np.sqrt(np.mean(np.square(amp), axis=-1) + eps)
    else:
        raise TypeError(f"argument 'amp' must be torch.Tensor or np.ndarray. got: {type(amp)}")

def rms_normalize(wav, ref_dB=-23.0, eps=1e-5):
    # RMS normalize
    rms = cal_rms(wav)
    ref_linear = np.power(10, ref_dB/20.)
    gain = ref_linear / (rms + eps)
    wav = gain * wav
    return gain, wav

def mel_basis(sr, n_fft, n_mel):
    basis = librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mel,fmin=0,fmax=sr//2,norm=1)
    return torch.from_numpy(basis)

def SNR(preds, target, zero_mean=False, reduce_batch=True):
    eps = torch.finfo(preds.dtype).eps

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    noise = target - preds

    val = (torch.sum(target**2, dim=-1) + eps) / (torch.sum(noise**2, dim=-1) + eps)
    val = 10 * torch.log10(val)

    return val.mean(0) if reduce_batch else val

def SI_SNR(preds, target, zero_mean=True, reduce_batch=True):
    eps = torch.finfo(preds.dtype).eps
    if preds.size(1) > 1:
        preds = rearrange(preds, 'b (c1 c2) t -> (b c1) c2 t', c2=1)
        target = rearrange(target, 'b (c1 c2) t -> (b c1) c2 t', c2=1)

    if zero_mean:
        target = target - torch.mean(target, dim=-1, keepdim=True)
        preds = preds - torch.mean(preds, dim=-1, keepdim=True)

    alpha = (torch.sum(preds * target, dim=-1, keepdim=True) + eps) \
          / (torch.sum(target**2, dim=-1, keepdim=True) + eps)
    target_scaled = alpha * target

    noise = target_scaled - preds

    val = (torch.sum(target_scaled**2, dim=-1) + eps)\
        / (torch.sum(noise**2, dim=-1) + eps)
    val = 10 * torch.log10(val)

    return val.mean(0) if reduce_batch else val

def SI_SDR(preds, target, reduce_batch=True):
    eps = torch.finfo(preds.dtype).eps
    if preds.size(1) > 1:
        preds = rearrange(preds, 'b (c1 c2) t -> (b c1) c2 t', c2=1)
        target = rearrange(target, 'b (c1 c2) t -> (b c1) c2 t', c2=1)

    alpha = ((preds * target).sum(-1, keepdims=True) + eps) \
          / ((target ** 2   ).sum(-1, keepdims=True) + eps)

    target_scaled = alpha * target
    noise = target_scaled - preds

    val = ((target_scaled ** 2).sum(-1) + eps) \
        / ((noise ** 2        ).sum(-1) + eps)
    val = 10 * val.log10()
    return val.mean(0) if reduce_batch else val

def spec_loss(est, tar, rescale = 10):
    est = est / rescale
    tar = tar / rescale
    loss = F.mse_loss(est, tar)
    return loss

def unfold(x, window_length, n_ch=1):
    x = rearrange(x, 'b c (t p) -> b c t p', p=window_length)
    n_frames = x.size(2)
    x = F.unfold(x, (n_frames-1,1))
    x = rearrange(x, 'b (c t) (p d) -> (b t) c (p d)', c=n_ch, t=n_frames-1, d=2, p=window_length)
    return x 


