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
import torchaudio.functional as TAF
import random
import math
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from dataset.loader import load_txt
from utils import unfold, rms_normalize
from networks.nfs import get_inverse_window

def process(x, p, sr, max_sec=30.):
    # trim to max sec
    if x.shape[0] > int(max_sec*sr):
        r = np.random.randint(x.shape[0]-int(max_sec*sr)+1)
        x = x[r:r+int(max_sec*sr)]
    # mixdown to mono
    if len(list(x.shape)) > 1:
        x = x.mean(-1)
    # resample to 48k
    x = torch.from_numpy(x).view(1,1,-1)
    if sr != 48000:
        x = TAF.resample(x, sr, 48000, resampling_method='kaiser_window')
    lens = int(x.size(-1) / 48000 * 120)

    if p.size(0) < lens:
        # x is longer
        x_lens = int(lens / 120 * 48000)
        i = np.random.randint(x.size(-1) - x_lens + 1)
        x = x.narrow(-1,i,x_lens)
    else:
        # p is longer
        i = np.random.randint(p.size(0) - lens + 1)
        p = p.narrow(0,i,lens)

    # trim to num of samples divisible by sr
    lens_sec = min(math.floor(x.size(-1) / 48000), math.floor(p.size(0) / 120))
    x = x.narrow(-1,0,int(lens_sec*48000))
    p = p.narrow(0,0,int(lens_sec*120))

    x = rms_normalize(x)[1]
    x = x.squeeze().numpy()    # (sec * 48000)
    p = p.squeeze().numpy()    # (sec * 120, 7)
    return x, p

def collect(data, txpos, root_dir, min_sec=3., max_sec=30.):
    for key in data.keys():
        iterator = tqdm(data[key])
        iterator.set_description(f"Prepare {key} with positions")
        for n, dp in enumerate(iterator):
            x, sr = sf.read(dp)
            if max(x.shape) < int(min_sec*sr):
                continue
            pos_path = txpos[np.random.randint(len(txpos))]
            p = load_txt(pos_path)    # (time, channel)
            x, p = process(x, p, sr, max_sec=max_sec)
            save_dir = os.path.join(root_dir, key)
            os.makedirs(save_dir, exist_ok=True)
            sf.write(f'{save_dir}/{n}.wav', x, samplerate=48000, subtype='PCM_16')
            np.savetxt(f'{save_dir}/{n}.txt', p)

if __name__ == '__main__':
    seed = 1234
    music_path   = '/data2/musan/music/*/*.wav'
    speech_path  = '/data2/musan/speech/*/*.wav'
    general_path = '/data2/musan/noise/*/*.wav'
    #vocal_path   = '/data/VocalSet/*/*/*/*.wav'
    vocal_path_1 = '/data/VocalSet/*/*/straight/*.wav'
    vocal_path_2 = '/data/VocalSet/*/*/vibrato/*.wav'
    txpos_path   = '/data2/binaural_dataset/binaural_dataset/testset/*/tx_positions.txt'
    np.random.seed(seed)
    random.seed(seed)
    root_dir = '/data2/binaural_dataset/modified/ood'
    min_sec = 5.
    max_sec = 30.

    music   = sorted(glob.glob(music_path))
    speech  = sorted(glob.glob(speech_path))
    general = sorted(glob.glob(general_path))
    #vocal   = sorted(glob.glob(vocal_path))
    vocal   = sorted(glob.glob(vocal_path_1))+sorted(glob.glob(vocal_path_2))
    txpos   = sorted(glob.glob(txpos_path))

    music   = random.choices(music,  k=30)
    speech  = random.choices(speech, k=30)
    general = random.choices(general,k=30)
    vocal   = random.choices(vocal,  k=30)

    data = {
        "speech": speech,
        "singing": vocal,
        "music": music,
        "general": general,
    }
    print([f"{key}: {len(data[key])}" for key in data.keys()])
    print(f"position: {len(txpos)}")
    print(f"trim into {max_sec} sec")
    collect(data, txpos, root_dir, min_sec, max_sec)

