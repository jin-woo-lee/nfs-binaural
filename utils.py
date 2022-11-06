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

def get_max_point(x):
    i = np.argmax(x)
    return i, x[i]

def save_samples(sample, directory, name, sr, b=0):
    dry_mag, wet_mag, est_mag = sample[0]
    dry_mel, wet_mel, est_mel = sample[1]
    dry_wav, wet_wav, est_wav = sample[2]
    #wet_gdm_lin, est_gdm_lin, wet_gds_lin, est_gds_lin = sample[3]
    #wet_gdm_mel, est_gdm_mel, wet_gds_mel, est_gds_mel = sample[4]
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
    #dr  = librosa.display.specshow(dry_mel[b,0], cmap='magma', ax=ax[0,1])
    pe1 = librosa.display.specshow(pe1,          cmap='bwr',   ax=ax[0,1])
    egl = librosa.display.specshow(est_mag[b,0], cmap='magma', ax=ax[1,1])
    egr = librosa.display.specshow(est_mag[b,1], cmap='magma', ax=ax[2,1])
    eml = librosa.display.specshow(est_mel[b,0], cmap='magma', ax=ax[3,1])
    emr = librosa.display.specshow(est_mel[b,1], cmap='magma', ax=ax[4,1])
    #---------- 
    dif_dry = dry_mag[b,0] - wet_mag[b,0]
    dif_mag = wet_mag - est_mag; dif_mel = wet_mel - est_mel
    #ddl = librosa.display.specshow(dif_dry,      cmap='bwr', ax=ax[0,2])
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
    #dif_wav = wet_wav - drf_wav; p = 10000; q = 10200
    #ax[6,0].plot(wet_wav[b,0,p:q], 'g-')
    #ax[6,0].plot(drf_wav[b,0,p:q], 'k-'); ax[6,0].axhline(y=0, color='k', lw=.3)
    #ax[6,1].plot(wet_wav[b,1,p:q], 'g-')
    #ax[6,1].plot(drf_wav[b,1,p:q], 'k-'); ax[6,1].axhline(y=0, color='k', lw=.3)
    #ax[6,2].plot(dif_wav[b,0,p:q], 'b-')
    #ax[6,2].plot(dif_wav[b,1,p:q], 'r-'); ax[6,2].axhline(y=0, color='k', lw=.3)
    #---------- 
    #ax[0,0].set_title("dry mag");      ax[0,1].set_title("dry mel")
    ax[0,0].set_title("dry mag");      ax[0,1].set_title("pos enc")
    ax[1,0].set_title("wet mag L");    ax[1,1].set_title("est mag L")
    ax[2,0].set_title("wet mag R");    ax[2,1].set_title("est mag R")
    ax[3,0].set_title("wet mel L");    ax[3,1].set_title("est mel L")
    ax[4,0].set_title("wet mel R");    ax[4,1].set_title("est mel R")
    #ax[0,2].set_title("mono - wet L")
    ax[0,2].set_title("pos enc")
    ax[1,2].set_title("diff mag L")
    ax[2,2].set_title("diff mag R")
    ax[3,2].set_title("diff mel L")
    ax[4,2].set_title("diff mel R")
    #plt.colorbar(dl,  ax=ax[0,0]); plt.colorbar(dr,  ax=ax[0,1])
    plt.colorbar(dl,  ax=ax[0,0]); plt.colorbar(pe1, ax=ax[0,1])
    plt.colorbar(wgl, ax=ax[1,0]); plt.colorbar(egl, ax=ax[1,1])
    plt.colorbar(wgr, ax=ax[2,0]); plt.colorbar(egr, ax=ax[2,1])
    plt.colorbar(wml, ax=ax[3,0]); plt.colorbar(eml, ax=ax[3,1])
    plt.colorbar(wmr, ax=ax[4,0]); plt.colorbar(emr, ax=ax[4,1])
    #plt.colorbar(ddl, ax=ax[0,2])
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
    dl.set_clim(dmin, dmax);#    dr.set_clim(dmin, dmax)
    wgl.set_clim(wgmin, wgmax); egl.set_clim(wgmin, wgmax)
    wgr.set_clim(wgmin, wgmax); egr.set_clim(wgmin, wgmax)
    wml.set_clim(wmmin, wmmax); eml.set_clim(wmmin, wmmax)
    wmr.set_clim(wmmin, wmmax); emr.set_clim(wmmin, wmmax)
    #ddl.set_clim(-40, 40)
    dgl.set_clim(-40, 40); dgr.set_clim(-40, 40)
    dml.set_clim(-40, 40); dmr.set_clim(-40, 40)
    pe1.set_clim(-2, 2); pe2.set_clim(-2, 2)
    #---------- 
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{name}-{b}-gen.png")
    plt.close()

    #dif_gdm_lin = wet_gdm_lin - est_gdm_lin; dif_gds_lin = wet_gds_lin - est_gds_lin
    #dif_gdm_mel = wet_gdm_mel - est_gdm_mel; dif_gds_mel = wet_gds_mel - est_gds_mel
    #dif_gdm_lin = np.absolute(np.absolute(dif_gdm_lin + np.pi) - 2*np.pi) - np.pi
    #dif_gdm_mel = np.absolute(np.absolute(dif_gdm_mel + np.pi) - 2*np.pi) - np.pi
    #fig2, ax = plt.subplots(8, 3, figsize=(10,20))
    ##---------- 
    #wgml = librosa.display.specshow(wet_gdm_lin[b,0],  cmap='Blues', ax=ax[0,0])
    #wgmr = librosa.display.specshow(wet_gdm_lin[b,1],  cmap='Blues', ax=ax[1,0])
    #wgsl = librosa.display.specshow(wet_gds_lin[b,0],  cmap='Blues', ax=ax[2,0])
    #wgsr = librosa.display.specshow(wet_gds_lin[b,1],  cmap='Blues', ax=ax[3,0])
    ##---------- 
    #egml = librosa.display.specshow(est_gdm_lin[b,0],  cmap='Blues', ax=ax[0,1])
    #egmr = librosa.display.specshow(est_gdm_lin[b,1],  cmap='Blues', ax=ax[1,1])
    #egsl = librosa.display.specshow(est_gds_lin[b,0],  cmap='Blues', ax=ax[2,1])
    #egsr = librosa.display.specshow(est_gds_lin[b,1],  cmap='Blues', ax=ax[3,1])
    ##---------- 
    #dgml = librosa.display.specshow(dif_gdm_lin[b,0],  cmap='bwr', ax=ax[0,2])
    #dgmr = librosa.display.specshow(dif_gdm_lin[b,1],  cmap='bwr', ax=ax[1,2])
    #dgsl = librosa.display.specshow(dif_gds_lin[b,0],  cmap='bwr', ax=ax[2,2])
    #dgsr = librosa.display.specshow(dif_gds_lin[b,1],  cmap='bwr', ax=ax[3,2])
    ##---------- 
    #wmml = librosa.display.specshow(wet_gdm_mel[b,0],  cmap='Blues', ax=ax[4,0])
    #wmmr = librosa.display.specshow(wet_gdm_mel[b,1],  cmap='Blues', ax=ax[5,0])
    #wmsl = librosa.display.specshow(wet_gds_mel[b,0],  cmap='Blues', ax=ax[6,0])
    #wmsr = librosa.display.specshow(wet_gds_mel[b,1],  cmap='Blues', ax=ax[7,0])
    ##---------- 
    #emml = librosa.display.specshow(est_gdm_mel[b,0],  cmap='Blues', ax=ax[4,1])
    #emmr = librosa.display.specshow(est_gdm_mel[b,1],  cmap='Blues', ax=ax[5,1])
    #emsl = librosa.display.specshow(est_gds_mel[b,0],  cmap='Blues', ax=ax[6,1])
    #emsr = librosa.display.specshow(est_gds_mel[b,1],  cmap='Blues', ax=ax[7,1])
    ##---------- 
    #dmml = librosa.display.specshow(dif_gdm_mel[b,0],  cmap='bwr', ax=ax[4,2])
    #dmmr = librosa.display.specshow(dif_gdm_mel[b,1],  cmap='bwr', ax=ax[5,2])
    #dmsl = librosa.display.specshow(dif_gds_mel[b,0],  cmap='bwr', ax=ax[6,2])
    #dmsr = librosa.display.specshow(dif_gds_mel[b,1],  cmap='bwr', ax=ax[7,2])
    ##---------- 
    #ax[0,0].set_title("gd mean wet L"); ax[0,1].set_title("gd mean est L")
    #ax[1,0].set_title("gd mean wet R"); ax[1,1].set_title("gd mean est R")
    #ax[2,0].set_title("gd std  wet L"); ax[2,1].set_title("gd std  est L")
    #ax[3,0].set_title("gd std  wet R"); ax[3,1].set_title("gd std  est R")
    #ax[0,2].set_title("gd mean diff L")
    #ax[1,2].set_title("gd mean diff R")
    #ax[2,2].set_title("gd std  diff L")
    #ax[3,2].set_title("gd std  diff R")
    #plt.colorbar(wgml,ax=ax[0,0]); plt.colorbar(egml,ax=ax[0,1])
    #plt.colorbar(wgmr,ax=ax[1,0]); plt.colorbar(egmr,ax=ax[1,1])
    #plt.colorbar(wgsl,ax=ax[2,0]); plt.colorbar(egsl,ax=ax[2,1])
    #plt.colorbar(wgsr,ax=ax[3,0]); plt.colorbar(egsr,ax=ax[3,1])
    #wgml.set_clim(0, np.pi); egml.set_clim(0, np.pi)
    #wgmr.set_clim(0, np.pi); egmr.set_clim(0, np.pi)
    ##wgml.set_clim(-np.pi, np.pi); egml.set_clim(-np.pi, np.pi)
    ##wgmr.set_clim(-np.pi, np.pi); egmr.set_clim(-np.pi, np.pi)
    #wgsl.set_clim(0, np.pi); egsl.set_clim(0, np.pi)
    #wgsr.set_clim(0, np.pi); egsr.set_clim(0, np.pi)
    #plt.colorbar(dgml,ax=ax[0,2]); dgml.set_clim(-np.pi, np.pi)
    #plt.colorbar(dgmr,ax=ax[1,2]); dgmr.set_clim(-np.pi, np.pi)
    #plt.colorbar(dgsl,ax=ax[2,2]); dgsl.set_clim(-np.pi, np.pi)
    #plt.colorbar(dgsr,ax=ax[3,2]); dgsr.set_clim(-np.pi, np.pi)
    ##---------- 
    #ax[4,0].set_title("gd mean wet L"); ax[4,1].set_title("gd mean est L")
    #ax[5,0].set_title("gd mean wet R"); ax[5,1].set_title("gd mean est R")
    #ax[6,0].set_title("gd std  wet L"); ax[6,1].set_title("gd std  est L")
    #ax[7,0].set_title("gd std  wet R"); ax[7,1].set_title("gd std  est R")
    #ax[4,2].set_title("gd mean diff L")
    #ax[5,2].set_title("gd mean diff R")
    #ax[6,2].set_title("gd std  diff L")
    #ax[7,2].set_title("gd std  diff R")
    #plt.colorbar(wmml,ax=ax[4,0]); plt.colorbar(emml,ax=ax[4,1])
    #plt.colorbar(wmmr,ax=ax[5,0]); plt.colorbar(emmr,ax=ax[5,1])
    #plt.colorbar(wmsl,ax=ax[6,0]); plt.colorbar(emsl,ax=ax[6,1])
    #plt.colorbar(wmsr,ax=ax[7,0]); plt.colorbar(emsr,ax=ax[7,1])
    #wmml.set_clim(0, np.pi); emml.set_clim(0, np.pi)
    #wmmr.set_clim(0, np.pi); emmr.set_clim(0, np.pi)
    #wmsl.set_clim(0, np.pi); emsl.set_clim(0, np.pi)
    #wmsr.set_clim(0, np.pi); emsr.set_clim(0, np.pi)
    #plt.colorbar(dmml,ax=ax[4,2]); dmml.set_clim(-np.pi, np.pi)
    #plt.colorbar(dmmr,ax=ax[5,2]); dmmr.set_clim(-np.pi, np.pi)
    #plt.colorbar(dmsl,ax=ax[6,2]); dmsl.set_clim(-np.pi, np.pi)
    #plt.colorbar(dmsr,ax=ax[7,2]); dmsr.set_clim(-np.pi, np.pi)
    ##---------- 

    #fig2.tight_layout()
    #fig2.savefig(f"{directory}/{name}-{b}-gds.png")
    #plt.close()


    #figs = [fig1, fig2]
    figs = [fig1]

    dry_wav = np.transpose(dry_wav[b])
    wet_wav = np.transpose(wet_wav[b])
    est_wav = np.transpose(est_wav[b])
    #drf_wav = np.transpose(drf_wav[b])
    sf.write(f"{directory}/{name}-{b}-dry.wav", dry_wav, samplerate=sr,subtype='PCM_16')
    sf.write(f"{directory}/{name}-{b}-wet.wav", wet_wav, samplerate=sr,subtype='PCM_16')
    sf.write(f"{directory}/{name}-{b}-est.wav", est_wav, samplerate=sr,subtype='PCM_16')
    #sf.write(f"{directory}/{name}-{b}-drf.wav", drf_wav, samplerate=sr,subtype='PCM_16')

    wavs = [dry_wav, wet_wav, est_wav]

    return figs, wavs

def save_planes(sample, directory, name, sr, bh=0, bf=0):
    #hor_ir, hor_tf, h2v_h, h2s_h, ive_h, ise_h = sample[0]
    #fro_ir, fro_tf, h2v_f, h2s_f, ive_f, ise_f = sample[1]
    hor_ir, hor_tf = sample[0]
    fro_ir, fro_tf = sample[1]

    fig1, ax = plt.subplots(2, 4, figsize=(7,10))
    #---------- 
    ir_hl = librosa.display.specshow(hor_ir[:,0,:].T, cmap='bwr', ax=ax[0,0])
    ir_hr = librosa.display.specshow(hor_ir[:,1,:].T, cmap='bwr', ax=ax[0,1])
    tf_hl = librosa.display.specshow(hor_tf[:,0,:].T, cmap='bwr', ax=ax[0,2])
    tf_hr = librosa.display.specshow(hor_tf[:,1,:].T, cmap='bwr', ax=ax[0,3])
    ax[0,0].set_title("hor IR L"); ax[0,1].set_title("hor IR R")
    ax[0,2].set_title("hor TF L"); ax[0,3].set_title("hor TF R")
    ir_hl.set_clim([-2,  +2]); ir_hr.set_clim([-2,  +2])
    tf_hl.set_clim([-30, 20]); tf_hr.set_clim([-30, 20])
    #---------- 
    ir_fl = librosa.display.specshow(fro_ir[:,0,:].T, cmap='bwr', ax=ax[1,0])
    ir_fr = librosa.display.specshow(fro_ir[:,1,:].T, cmap='bwr', ax=ax[1,1])
    tf_fl = librosa.display.specshow(fro_tf[:,0,:].T, cmap='bwr', ax=ax[1,2])
    tf_fr = librosa.display.specshow(fro_tf[:,1,:].T, cmap='bwr', ax=ax[1,3])
    ax[1,0].set_title("fro IR L"); ax[1,1].set_title("fro IR R")
    ax[1,2].set_title("fro TF L"); ax[1,3].set_title("fro TF R")
    ir_fl.set_clim([-2,  +2]); ir_fr.set_clim([-2,  +2])
    tf_fl.set_clim([-30, 20]); tf_fr.set_clim([-30, 20])
    #---------- 
    fig1.tight_layout()
    fig1.savefig(f"{directory}/{name}-gen.png")
    plt.close()

    #vh_max = np.max(ive_h); vh_min = np.min(ive_h)
    #sh_max = np.max(ise_h); sh_min = np.min(ise_h)
    #vf_max = np.max(ive_f); vf_min = np.min(ive_f)
    #sf_max = np.max(ise_f); sf_min = np.min(ise_f)
    #assert sf_max == sh_max, f"{sf_max} == {sh_max}"
    #assert sf_min == sh_min, f"{sf_min} == {sh_min}"
    #fig2, ax = plt.subplots(2, 4, figsize=(7,10))
    ##---------- 
    #fig_00 = librosa.display.specshow(h2v_h[:,0,:].T, cmap='bwr', ax=ax[0,0])
    #fig_01 = librosa.display.specshow(ive_h[:,0,:].T, cmap='bwr', ax=ax[0,1])
    #fig_02 = librosa.display.specshow(h2s_h[:,0,:].T, cmap='bwr', ax=ax[0,2])
    #fig_03 = librosa.display.specshow(ise_h[:,0,:].T, cmap='bwr', ax=ax[0,3])
    #fig_10 = librosa.display.specshow(h2v_f[:,0,:].T, cmap='bwr', ax=ax[1,0])
    #fig_11 = librosa.display.specshow(ive_f[:,0,:].T, cmap='bwr', ax=ax[1,1])
    #fig_12 = librosa.display.specshow(h2s_f[:,0,:].T, cmap='bwr', ax=ax[1,2])
    #fig_13 = librosa.display.specshow(ise_f[:,0,:].T, cmap='bwr', ax=ax[1,3])
    #ax[0,0].set_title("estim hor view"); ax[0,2].set_title("estim hor subj")
    #ax[1,0].set_title("estim fro view"); ax[1,2].set_title("estim fro subj")
    #ax[0,1].set_title("input hor view"); ax[0,3].set_title("input hor subj")
    #ax[1,1].set_title("input fro view"); ax[1,3].set_title("input fro subj")
    #fig_00.set_clim([vh_min, vh_max]); fig_01.set_clim([sh_min, sh_max])
    #fig_10.set_clim([vf_min, vf_max]); fig_11.set_clim([sf_min, sf_max])
    #fig_02.set_clim([vh_min, vh_max]); fig_03.set_clim([sh_min, sh_max])
    #fig_12.set_clim([vf_min, vf_max]); fig_13.set_clim([sf_min, sf_max])
    ##---------- 
    #fig2.tight_layout()
    #fig2.savefig(f"{directory}/{name}-cls.png")
    #plt.close()

    #figs = [fig1, fig2]
    figs = [fig1]

    hor_ir = np.transpose(hor_ir[bh]); fro_ir = np.transpose(fro_ir[bf])
    sf.write(f"{directory}/{name}-{bh}-hor.wav", hor_ir, samplerate=sr, subtype='PCM_16')
    sf.write(f"{directory}/{name}-{bf}-fro.wav", fro_ir, samplerate=sr, subtype='PCM_16')
    wavs = [hor_ir, fro_ir]

    return figs, wavs


def chunk_sequence(sequences, lens):
    for i, data in enumerate(sequences):
        r = lens - data.size(-1) % lens
        data = F.pad(data, (0,r))
        data = data.reshape(-1, 1, lens)
        out = data if i == 0 else torch.cat((out,data), dim=0)
    return out

def pad_sequence(sequences):
    '''
    pad sequence to same length (max length)
    ------------------
    input:
        sequences --- a list of tensor with variable length
        pad_mode  --- padding option in ['zeros', 'tile']
    return:
        a tensor with max length
    '''
    lengths = [data.size(-1) for data in sequences]
    max_lens = max(lengths)

    batch_size = len(sequences)
    trailing_dims = sequences[0].size()[1:]
    out_dims = (batch_size, max_lens) + trailing_dims
    dtype = sequences[0].data.type()
    out = torch.zeros(*out_dims).type(dtype)
    for i, data in enumerate(sequences):
        out[i] = data.tile(max_lens // lengths[i] + 1).narrow(-1,0,max_lens)
    return out, lengths

def pad_batch(batch):
    hrir = [d['hrir'] for d in batch]
    view = [d['view'] for d in batch]

    hrir = torch.cat(hrir, dim=0)
    view = torch.cat(view, dim=0)

    return {
        'hrir': hrir,   # (B, N, 2, 256)
        'view': view,   # (B, N, 2, 1)
    }

def split_nbhd(x, factor=1):
    #m = x.size(1) * factor - (factor-1)
    #x = rearrange(x, 'b n c t -> b c t n')
    #x = F.interpolate(
    #    x, size=(x.size(2), m),
    #    mode='bilinear', antialias=True,
    #    align_corners=True)
    #x = rearrange(x, 'b c t n -> b n c t')
    ref = x.narrow(1,0,1).tile(1,x.size(1)-1,1,1)
    tar = x.narrow(1,1,x.size(1)-1)
    ref = rearrange(ref, 'b n c t -> (b n) c t')
    tar = rearrange(tar, 'b n c t -> (b n) c t')
    return ref, tar

def chunk_batch(batch, lens, orig_sr, targ_sr):
    input = [d['input'] for d in batch]
    rate  = [d['rates'] for d in batch]

    input = chunk_sequence(input, lens)
    #rm = 'sinc_interpolation'
    rm = 'kaiser_window'
    downs = TAF.resample(input, orig_sr, targ_sr, resampling_method=rm)  # downsample
    downs = TAF.resample(downs, targ_sr, orig_sr, resampling_method=rm)  # upsample
    rate = [targ_sr] * input.size(0)

    return {
        'input': input,   # (B, lens)
        'downs': downs,   # (B, lens)
        'rates': rate,    # (B,)
    }

def set_lengths(wav_list, lens):
    outdims = (len(wav_list), 1, lens)
    out = torch.zeros(*outdims).type(wav_list[0].type())
    for i, x in enumerate(wav_list):
        if x.size(-1) > lens:
            rnd = np.random.randint(x.size(-1) - lens)
            x = x.narrow(-1,rnd,lens)
        else:
            res = lens - x.size(-1)
            rep = res // x.size(-1) + 2
            x = x.tile(rep).narrow(-1,0,lens)
        out[i] = x
    return out

def rate_aug(x, sr_list):
    lens = x.size(-1)
    y = torch.zeros_like(x)
    for i, sr in enumerate(sr_list):
        #------------------------------ 
        #scale = np.random.randint(2,7)
        #rate = sr // scale
        #------------------------------ 
        den = np.random.randint(2,7)
        num = np.random.randint(1,den)
        rate = sr * num // den
        #------------------------------ 
        #rate = sr // 4
        #------------------------------ 
        #rm = 'sinc_interpolation'
        rm = 'kaiser_window'
        chunk = TAF.resample(x[i],  sr, rate, resampling_method=rm)  # downsample
        #++++++++++++++++++++++++++++++ 
        chunk = TAF.resample(chunk, rate, sr, resampling_method=rm)  # upsample
        chunk = chunk.narrow(-1,0,lens)        # set length
        #++++++++++++++++++++++++++++++ 
        y[i,0,:chunk.size(-1)] = chunk[0]
        sr_list[i] = rate
    #++++++++++++++++++++++++++++++ 
    y = y.narrow(-1,0,chunk.size(-1))        # set length
    #++++++++++++++++++++++++++++++ 
    return y, sr_list


def speed_aug(wav_list, sr, low=0.95, high=1.5):
    spd = []
    for i, waveform in enumerate(wav_list):
        if np.random.random() > 0.5:
            speed = (high-low) * np.random.random() + low
            effects = [
                ["speed", f"{speed}"],
                ["rate", f"{sr}"],
            ]
            waveform = waveform.unsqueeze(0)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sr, effects)
            waveform = waveform.squeeze(0)
            wav_list[i] = waveform
            spd.append(speed)
        else:
            spd.append(1.)
    return wav_list, spd

codec_configs = [
    ({"format": "wav", "encoding": 'ULAW', "bits_per_sample": 8}, "8 bit mu-law"),
    ({"format": "gsm"}, "GSM-FR"),
    ({"format": "mp3", "compression": -9}, "MP3"),
    ({"format": "vorbis", "compression": -1}, "Vorbis"),
]
def codec_aug(wav_list, sr):
    cdc = []
    for i, waveform in enumerate(wav_list):
        r = np.random.randint(len(codec_configs) + 1)
        if r < len(codec_configs):
            param, c = codec_configs[r]
            waveform = waveform.unsqueeze(0)
            waveform = TAF.apply_codec(waveform, sr, **param)
            wav_list[i] = waveform.squeeze(0)
            cdc.append(c)
        else:
            cdc.append("original")
    return wav_list, cdc


def filter_conv_dict(dicts):
    filtered_dicts = {}
    for key in dicts.keys():
        new_key = key

        if 'feature_extractor' in key:
            if len(key.split('.')) == 5:
                "feature_extractor.conv_layers.0.0.weight",
                _, conv_layers, i, _, wb = key.split('.')
                new_key = '.'.join([conv_layers, i, 'conv', wb])
            else:
                "feature_extractor.conv_layers.0.2.1.weight",
                _, conv_layers, i, _, _, wb = key.split('.')
                new_key = '.'.join([conv_layers, i, 'norm', wb])
            filtered_dicts[new_key] = dicts[key]

    return filtered_dicts


def filter_dict(dicts, selective=False):
    filtered_dicts = {}
    for key in dicts.keys():
        new_key = key
        if selective and ('post_extract_proj' in key):
            continue

        if 'feature_extractor' in key:
            if len(key.split('.')) == 5:
                "feature_extractor.conv_layers.0.0.weight",
                feature_extractor, conv_layers, i, _, wb = key.split('.')
                new_key = '.'.join([feature_extractor, conv_layers, i, 'conv', wb])
            else:
                if selective:
                    continue
                "feature_extractor.conv_layers.0.2.1.weight",
                feature_extractor, conv_layers, i, _, _, wb = key.split('.')
                new_key = '.'.join([feature_extractor, conv_layers, i, 'norm', wb])
        elif 'encoder' in key:
            if 'pos_conv' in key:
                "encoder.pos_conv.0.bias"
                encoder, pos_conv, i, wb = key.split('.')
                new_key = '.'.join([encoder, pos_conv, 'conv', wb])
        filtered_dicts[new_key] = dicts[key]

    del filtered_dicts["mask_emb"]
    del filtered_dicts["quantizer.vars"]
    del filtered_dicts["quantizer.weight_proj.weight"]
    del filtered_dicts["quantizer.weight_proj.bias"]
    del filtered_dicts["project_q.weight"]
    del filtered_dicts["project_q.bias"]
    del filtered_dicts["final_proj.weight"]
    del filtered_dicts["final_proj.bias"]
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

def adjust_noise(source, noise, snr):
    if isinstance(source, torch.Tensor):
        eps = torch.finfo(source.dtype).eps
    else:
        eps = np.finfo(np.float32).eps
    noise_rms = cal_rms(noise) # noise rms

    num = cal_rms(source) # source rms
    den = 10. ** (snr/20)
    desired_noise_rms = num / den

    # calculate gain
    gain = desired_noise_rms / (noise_rms + eps)
    noise = gain * noise

    return source + noise


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def chunk_wav(x, win_length):
    B, C, T = x.shape
    residual = win_length - (T - (T // win_length) * win_length)
    residual = residual % win_length
    x_pad = F.pad(x, (0,residual))
    return rearrange(x_pad, 'b c (h w) -> b c h w', w=win_length)

def get_mean_std(x, win_length=8, upsample=False, numpy=False):
    chunk = chunk_wav(x, win_length)
    mean = chunk.mean(-1)
    stds = chunk.std(-1)
    if upsample:
        mean = F.interpolate(mean, size=x.size(-1))
        stds = F.interpolate(stds, size=x.size(-1))
    if numpy:
        mean = mean.detach().cpu().numpy()
        stds = stds.detach().cpu().numpy()
    return mean, stds


def stats_loss(est, tar, win_length=8):
    si_est = F.instance_norm(est)
    si_tar = F.instance_norm(tar)

    si_est_mean, si_est_std = get_mean_std(si_est, win_length)
    si_tar_mean, si_tar_std = get_mean_std(si_tar, win_length)
    si_est_abm = si_est_mean.abs()
    si_tar_abm = si_tar_mean.abs()

    si_abm_loss = F.l1_loss(si_est_abm, si_tar_abm)
    si_std_loss = F.l1_loss(si_est_std, si_tar_std)
    return si_abm_loss + si_std_loss

def batched_cross_correlation(x, y):
    assert x.size(1) == 1 and y.size(1) == 2, f"x: {x.shape} y: {y.shape}"
    y = F.pad(y, pad=(0,y.size(2)-1), mode='constant')
    yl, yr = y.chunk(2, 1)
    corr_l = rearrange(F.conv1d(yl,x).diagonal(0), 't (b c) -> b c t', c=1)
    corr_r = rearrange(F.conv1d(yr,x).diagonal(0), 't (b c) -> b c t', c=1)

    logit = torch.cat((corr_l, corr_r), dim=1)
    label = logit.argmax(-1)
    return logit, label

def framewise_cross_correlation(dry, wet, win_length=400):
    bz = dry.size(0)

    chunk_d = chunk_wav(dry, win_length)    # (batch, 1, frames, window)
    chunk_w = chunk_wav(wet, win_length)    # (batch, 2, frames, window)
    chunk_d = rearrange(chunk_d, 'b c f w -> (b f) c w')
    chunk_w = rearrange(chunk_w, 'b c f w -> (b f) c w')
    logit, label = batched_cross_correlation(chunk_d, chunk_w)
    logit = rearrange(logit, '(b f) c w -> b c f w', b=bz)
    label = rearrange(label, '(b f) c -> b c f', b=bz)

    threshold = 0.05
    mask = logit.std(-1).ge(threshold)
    label = mask * label
    return logit, label

def render_cross_correlation(dry, cor, win_length=400):
    bz = dry.size(0)

    dry = chunk_wav(dry, win_length); dry = rearrange(dry, 'b c f w -> (b f) c w')
    cor = rearrange(cor, 'b c f w -> (b f) c w')

    dry_hat = torch.fft.rfft(dry, dry.size(-1))
    cor_hat = torch.fft.rfft(cor, cor.size(-1))

    wet = torch.fft.irfft(dry_hat * cor_hat, dry.size(-1))
    wet = rearrange(wet, '(b f) c w -> b c (f w)', b=bz)

    return wet


def feature_loss(est, gt):
    '''
    est: output of conv layer [6,5,4,3,2,1]
    gt : output of conv layer [0,1,2,3,4,5,6]
    '''
    loss = 0
    rev = len(gt)-2
    for l in range(len(est)):
        loss += F.mse_loss(est[l], gt[rev-l])
    return loss


def diff(x, axis):
    """ Take the finite difference of a tensor along an axis.
    """
    size = x.size(axis) - 1
    slice_front = torch.narrow(x, axis, 1, size)
    slice_back = torch.narrow(x, axis, 0, size)
    return slice_front - slice_back

def unwrap(p, discont=np.pi, axis=-1):
    """ Unwrap a cyclical phase tensor.
    """
    dd = diff(p, axis=axis)
    ddmod = torch.fmod(dd+np.pi, 2.0*np.pi) - np.pi

    idx = torch.logical_and(torch.eq(ddmod, -np.pi),torch.gt(dd,0))
    ddmod = torch.where(idx, torch.ones_like(ddmod) *np.pi, ddmod)
    ph_correct = ddmod - dd
    
    idx = torch.le(torch.abs(dd), discont)
    ddmod = torch.where(idx, torch.zeros_like(ddmod), dd)
    ph_cumsum = torch.cumsum(ph_correct, dim=axis)
    
    pad = (1,0)
    if (axis != -1) or (axis != len(p.shape)):
        ph_cumsum = ph_cumsum.transpose(-1,axis)
    ph_cumsum = F.pad(ph_cumsum, pad)
    if (axis != -1) or (axis != len(p.shape)):
        ph_cumsum = ph_cumsum.transpose(-1,axis)
    unwrapped = p + ph_cumsum
    return unwrapped

def instantaneous_frequency(phase_angle, time_axis=-1):
    """ Transform a fft tensor from phase angle to instantaneous frequency.
    """
    phase_unwrapped = unwrap(phase_angle, axis=time_axis)
    dphase = diff(phase_unwrapped, axis=time_axis)
    
    iphase = torch.narrow(phase_unwrapped, time_axis, 0, 1)
    dphase = torch.cat([iphase, dphase], dim=time_axis) / np.pi
    return dphase

def group_delay(phase, freq_axis=-1):
    phase_unwrapped = unwrap(phase, axis=freq_axis)
    dphase = diff(phase_unwrapped, axis=freq_axis)
    dphase_fmod = torch.fmod(dphase + 2*np.pi, 2.0*np.pi) - np.pi
    return dphase_fmod

def mel_basis(sr, n_fft, n_mel):
    basis = librosa.filters.mel(sr=sr,n_fft=n_fft,n_mels=n_mel,fmin=0,fmax=sr//2,norm=1)
    return torch.from_numpy(basis)

def local_stats(gd, w):
    """ gd : (batch, channel, freq, time)
        w  : (mels, freq)
        mu : (batch, channel, mels, time)
        std: (batch, channel, mels, time)
    """
    mu  = torch.matmul(w, gd)
    #mu  = cplx_angle(torch.matmul(w + 1j * torch.zeros_like(w), torch.exp(1j * gd)))
    std = (torch.matmul(w, gd.pow(2)) + 1e-3).sqrt()
    return mu, std

def cplx_angle(cplx):
    re, im = torch.view_as_real(cplx).chunk(2, -1)    # 2 * (B, C, F, T, 1)
    ang = torch.atan2(im, re).squeeze(-1)             # (B, C, F, T)
    return ang

def freqz(b, a=1, N=512, whole=False):
    lastpoint = 2 * np.pi if whole else np.pi
    n_fft = N if whole else N * 2
    h = torch.fft.rfft(b, n=n_fft, axis=-1).narrow(-1,0,N)
    h /= a
    if whole:
        stop = -1 if n_fft % 2 == 1 else -2
        h_flip = torch.narrow(stop, 0, -1)
        h = torch.cat((h, h[h_flip].conj()))
    return h

def sample_loss(e_results, d_results, z_results):
    loss = 0
    N = len(e_results)
    for n in range(N):
        e = e_results[n]
        d = d_results[N-1-n]

        loss += F.binary_cross_entropy_with_logits(d, e.detach())
        loss += F.binary_cross_entropy_with_logits(e, d.detach())
    mu, logvar = z_results[-1]
    kld_ = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(kld_).mul_(-0.5)
    return {
        "loss": loss,
    }

def reparameterize(stats, size=None, axis=None):
    mu, logvar = stats
    std = logvar.mul(0.5).exp_()
    if size is not None:
        dims = [size if s == axis else 1 for s in range(len(mu.shape))]
        eps = std.tile(dims)
    else:
        eps = std
    eps = torch.randn_like(eps)
    return eps.mul(std).add_(mu)


def disc_loss(outputs, target, reduce_mean=True):
    assert target in [1, 0]
    loss = 0
    for disc in outputs:
        if reduce_mean:
            loss += torch.mean((target - disc)**2)
        else:
            loss += adv_loss(disc, target)
    return loss

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

def refresh_anchor(x, n, rate):
    if rate:
        assert isinstance(x,list), "provide list of encoder layer results"
        return True if n % rate == 0 else False
    else:
        return False


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

def LSD(pred_mag, targ_mag, reduce_batch=True):
    eps = torch.finfo(pred_mag.dtype).eps
    if pred_mag.size(1) > 1:
        pred_mag = rearrange(pred_mag, 'b (c1 c2) t -> (b c1) c2 t', c2=1)
        targ_mag = rearrange(targ_mag, 'b (c1 c2) t -> (b c1) c2 t', c2=1)

    pred_pow = 2 * pred_mag    # (B, F, T)
    targ_pow = 2 * targ_mag    # (B, F, T)

    val = (pred_pow - targ_pow).pow(2).mean(1).sqrt().mean(1)
    return val.mean(0) if reduce_batch else val



def randn_like_stats(x, win_length=256):
    #B, C, _, T = x.shape
    x = x.squeeze(2).permute(1,2,0)
    B, C, T = x.shape
    z = torch.randn_like(x)

    residual = win_length - (T - (T // win_length) * win_length)
    x = F.pad(x, (0,residual), mode='reflect')

    m = x.reshape(B,C,-1,win_length).mean(-1)
    s = x.reshape(B,C,-1,win_length).std(-1)

    m = F.interpolate(m, scale_factor=win_length)[:,:,:T]
    s = F.interpolate(s, scale_factor=win_length)[:,:,:T]

    z = m + z * s
    z = z.permute(2,0,1).unsqueeze(2)
    return z


#def sinusoidal_encoding(x):
#    bsz, freq, lens = x.shape
#    e = torch.zeros_like(x)
#    chan = freq // 2
#
#    c = torch.arange(chan, dtype=x.dtype, device=x.device).reshape((1,chan,1))
#    t = torch.arange(lens, dtype=x.dtype, device=x.device).reshape((1,1,lens))
#    w = (chan / 40)**(- 2*c / chan)
#    w = F.interpolate(w.transpose(2,1), scale_factor=2).transpose(2,1)
#    e[:,0::2,:] = torch.sin(w[:,0::2,:]*np.pi*t)
#    e[:,1::2,:] = torch.cos(w[:,1::2,:]*np.pi*t)
#    e = e.flip(1)
#    return e

def rate_encoding(rate, x):
    bsz, chan, lens = x.shape
    rate = [rate] * bsz if isinstance(rate, int) else rate
    c = [torch.linspace(0, r / 10, chan, dtype=x.dtype, device=x.device) for r in rate]
    c = torch.cat(c, dim=0)
    c = c.reshape(len(rate),chan,1).repeat(1,1,lens)
    w = 100**(- c / chan)
    e = torch.sin(w*np.pi*c)
    return e.flip(1)

def mask_by_rate(x, rate, ref_sr=48000):
    bsz, chan, lens = x.shape

    shift = torch.Tensor(rate).to(x.device).reshape(bsz,1,1)
    scale = ref_sr / 80

    fmask = torch.linspace(0,ref_sr,chan, dtype=x.dtype, device=x.device)
    fmask = (fmask.reshape(1,chan,1).repeat(bsz,1,1) - shift) / scale
    fmask = torch.sigmoid(fmask)

    return x * fmask


def magphase(x, n_fft=None):
    n_fft = n_fft if n_fft else x.size(-1)
    spec = torch.fft.rfft(x, n_fft)
    mag = spec.abs() + torch.finfo(x.dtype).eps
    phs = spec / mag
    return mag, phs

def get_abs_roll(x):
    return x.abs().argmax(-1, keepdim=True).long()


def diff_roll(x, n):
    """ x: (batch*time, channel, taps)
        n: (batch*time, channel, 1)
        -> (batch*time, channel, taps)
    """
    taps = x.size(-1)
    mag, phs = magphase(x)                # (batch*time, channel, freq)
    linspace = torch.linspace(0, - np.pi, taps // 2 + 1).to(x.device)

    phs_roll = (1j * n * linspace).exp()  # (batch*time, channel, freq)
    phs = phs * phs_roll
    return torch.fft.irfft(mag * phs, taps)

#def delay_loss(roll, drf, wet, taps, eps=1e-5):
#    """ roll: (batch*time, 2, 1)
#        drf : (batch, 2, taps*time)
#        wet : (batch, 2, taps*time)
#    """
#    drf = unfold(drf, taps // 2, n_ch=drf.size(1))    # (batch*time, 1, taps)
#    wet = unfold(wet, taps // 2, n_ch=wet.size(1))    # (batch*time, 2, taps)
#
#    drf = rearrange(drf, 'b c p -> (b c) p')
#    wet = rearrange(wet, 'b c p -> (b c) p')
#    #sim = 1 - F.cosine_similarity(drf, wet)
#    #sim = wet.pow(2).sum(-1).sqrt() * sim
#    #return sim.mean(0)
#    drf = (drf.abs() + eps).log()
#    wet = (wet.abs() + eps).log()
#    return F.mse_loss(drf, wet)

def consistency(roll, batch_size):
    """ roll: (batch*time, 2, 1)
    """
    roll = rearrange(roll, '(b t) c p -> b c (t p)', b=batch_size)
    devs = (roll - roll.mean(-1, keepdim=True)).pow(2).mean(-1).sqrt()
    return F.mse_loss(devs, torch.zeros_like(devs))


def latent_loss(l):
    z = torch.randn_like(l)
    est_m = l.mean([1,2]);  est_s = l.std([1,2])
    tar_m = z.mean([1,2]);  tar_s = z.std([1,2])
    return F.mse_loss(est_m, tar_m) + F.mse_loss(est_s, tar_s)

def ce_loss(logit, label):
    logit = rearrange(logit, 'b c t -> (b c) t')
    label = rearrange(label, 'b c t -> (b c t)').long()
    return F.cross_entropy(logit, label)

def pos_to_azim(x):
    """ (x y z qx qy qz qw) --> (azimuth)
    """
    zzz
    return az

def cls_loss(logit, label):
    """ logit: (batch, time, class)
        label: (batch, time)
    """
    logit = rearrange(logit, 'b t c -> (b t) c')
    label = rearrange(label, 'b t -> (b t)')
    return F.cross_entropy(logit, label)

def similarity(est, tar):
    est = rearrange(est, 'b t w -> (b t) w')
    tar = rearrange(tar, 'b t w -> (b t) w').detach()
    return 1 - F.cosine_similarity(est, tar).mean(0)

def spec_loss(est, tar, rescale = 10):
    est = est / rescale
    tar = tar / rescale
    loss = F.mse_loss(est, tar)
    return loss

def weighted_spec_loss(est_hrtf, ref_hrtf, est_view, ref_view, rescale = 10):
    est_hrtf = est_hrtf / rescale
    ref_hrtf = ref_hrtf / rescale
    hrtf_dist = - (est_hrtf - ref_hrtf).abs().mean([1,2]).log()
    view_dist = + (est_view - ref_view).pow(2).mean([1,2]).sqrt()
    weighted_dist = view_dist * hrtf_dist
    return weighted_dist.mean(0)

def sinusoidal_encoding(x, lens, ff=0.01, tf=1):
    eps = torch.finfo(x.dtype).eps
    x = x * ff
    t = torch.ones((x.size(0),x.size(1),lens), dtype=x.dtype,device=x.device).cumsum(-1)
    t = t * tf
    return torch.sin(x * t)

def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    return points_grad


def eikonal_loss(x, y):
    dydx = gradient(x, y)
    gnorm = dydx.norm(2, -1)
    return F.l1_loss(gnorm, torch.ones_like(gnorm))


def unfold(x, window_length, n_ch=1):
    x = rearrange(x, 'b c (t p) -> b c t p', p=window_length)
    n_frames = x.size(2)
    x = F.unfold(x, (n_frames-1,1))
    x = rearrange(x, 'b (c t) (p d) -> (b t) c (p d)', c=n_ch, t=n_frames-1, d=2, p=window_length)
    return x 

def get_gtcc(x, y, window_length=2000):
    assert x.size(1) == 1 and y.size(1) == 2, f"x:{x.shape}, y:{y.shape}"
    """ input  x: [ b, 1, (t p) ] 
        input  y: [ b, 2, (t p) ]
        output c: [ b, 2, t (p d) ] d-2
    """
    n_frames = x.size(-1) // window_length
    x = unfold(x, window_length, n_ch=1)
    y = unfold(y, window_length, n_ch=2)
    c = batched_cross_correlation(x, y)[0]
    c = rearrange(c, '(b t) c (p d) -> b c t (p d)', c=2, t=n_frames-1, d=2, p=window_length)
    return c, y

def ola_render(x, c, alpha, window_length=2000):
    assert x.size(1) == 1, f"x:{x.shape}"
    """ input  x: [ b, 1, (t p) ] 
               c: [ b, 2, t, (p d) ] d=2
        output z: [ b, 2, (t p) ]
    """
    n_frames = x.size(-1) // window_length
    x = unfold(x, window_length, n_ch=x.size(1))
    x = rearrange(x, '(b t) c p -> b c (t p)', t=n_frames-1, p=2*window_length)
    c = rearrange(c, 'b c t (p d) -> b c t (p d)', d=2, p=window_length)
    z = alpha * render_cross_correlation(x, c, 2*window_length)
    z = rearrange(z, 'b c (t p d) -> b (c t) (p d)', c=2, t=n_frames-1, d=2, p=window_length)
    
    window = torch.hann_window(2*window_length, periodic=True).to(z.device)
    z = z * rearrange(window, '(b c t p d) -> b (c t) (p d)', b=1, c=1, t=1, d=1)
    
    z = F.fold(z, (n_frames,window_length), (n_frames-1,1))
    z = rearrange(z, 'b c t p -> b c (t p)')
    return z

def polar(x, basis='10'):
    assert x.size(-1) == 2, x.shape        # (B C F T 2)
    assert basis in ['10', '01']
    #------------------------------ 
    eps = torch.finfo(x.dtype).eps; x = x + eps   # for numerical stability of atan2
    x = x / x.pow(2).sum(-1, keepdim=True).pow(.5)
    #------------------------------ 
    re, im = x.chunk(2, -1)                # 2 * (B C F T 1)
    if basis == '10':
        x = torch.atan2(im, re).squeeze(-1)    # (B C F T)
    if basis == '01':
        x = torch.atan2(re, im).squeeze(-1)    # (B C F T)
    return x

def polar_loss(est, tar):
    diff = est - tar
    diff = ((diff + np.pi).abs() - 2*np.pi).abs() - np.pi
    return F.mse_loss(diff, torch.zeros_like(diff))

def symlog(x, shift=0.2):
    s = torch.sign(x); y = shift * torch.ones_like(x)
    return s * (s * x + shift).log() - s * y.log()

def symlog_loss(est, tar):
    est = symlog(est); tar = symlog(tar)
    return F.l1_loss(est, tar)

#def gd_loss(est_gd, tar_gd, est_mag, tar_mag, criterion, ignore_below=0.1):
def gd_loss(est_gd, tar_gd, criterion):
    #est_mag = est_mag.narrow(2,1,est_gd.size(2))
    #tar_mag = tar_mag.narrow(2,1,tar_gd.size(2))
    #est_gd = est_gd.flatten(); est_mag = torch.pow(10, est_mag / 20).flatten()
    #tar_gd = tar_gd.flatten(); tar_mag = torch.pow(10, tar_mag / 20).flatten()
    #est_mask = est_mag > ignore_below * torch.mean(tar_mag)
    #tar_mask = tar_mag > ignore_below * torch.mean(tar_mag)
    #indices = torch.nonzero(tar_mask * est_mask).view(-1)
    #est_gd = torch.index_select(est_gd, 0, indices)
    #tar_gd = torch.index_select(tar_gd, 0, indices)
    #------------------------------ 
    #est_gd = torch.exp(1j * est_gd);     tar_gd = torch.exp(1j * tar_gd)
    #est_gd = torch.view_as_real(est_gd); tar_gd = torch.view_as_real(tar_gd)
    #------------------------------ 
    loss = criterion(est_gd, tar_gd)
    #loss = loss * (- loss).exp().detach()
    return loss


def preemphasis(audio, pre=0.97):
    x1 = audio.narrow(-1,0,1); n = audio.size(-1)-1
    x2 = audio.narrow(-1,1,n) - pre * audio.narrow(-1,0,n)
    return torch.cat((x1, x2), dim=-1)

def dly_loss(est_dly, tar_dly, tar_wav, ignore_below=0.1):
    tar_wav = rearrange(tar_wav, '(b t) c p -> b c t p', b=tar_dly.size(0))
    tar_mag = tar_wav.pow(2).mean(-1).sqrt().flatten()

    est_dly = est_dly.flatten()
    tar_dly = tar_dly.flatten()

    dly_mask = (tar_dly < 5) * (tar_dly > 2.5)
    mag_mask = tar_mag > ignore_below * torch.mean(tar_mag)

    indices = torch.nonzero(dly_mask * mag_mask).view(-1)

    est_dly = torch.index_select(est_dly, 0, indices)
    tar_dly = torch.index_select(tar_dly, 0, indices)

    loss = F.l1_loss(est_dly, tar_dly)
    return loss

def lowcut_renormalize(x, sr, cut=True):
    x = TAF.highpass_biquad(x, sr, cutoff_freq=20, Q=0.1) if cut else x
    x = rms_normalize(x)[1]
    return x


