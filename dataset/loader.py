import torch
import numpy as np 
from torch.utils import data
from torch.utils.data import DataLoader
import os 
import soundfile as sf 
import pickle
import glob
import scipy
import scipy.io
import random
import librosa
from tqdm import tqdm
import json
import time
import torchaudio.functional as TAF

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import *

def load_wav(path):
    x, sr = sf.read(path)
    return torch.from_numpy(x)

def load_txt(path):
    def str2list(s):
        return [float(i) for i in s.strip().split(' ')]
    with open(path, "r") as f:
        data = f.readlines()
        data = list(map(lambda s: str2list(s), data))
    return torch.tensor(data)

class GenericDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            mode='Train',
            lens_sec=1.,
        ):
        np.random.seed(0)
        mode = mode.lower()
        self.sr = 48000

        self.mode = mode
        root_dir = '/data2/binaural_dataset/binaural_dataset'
        subset_dir = os.path.join(root_dir, 'trainset' if mode=='train' else 'testset')

        if mode == 'valid':
            dry_list = glob.glob(f"{subset_dir}/validation_sequence/mono.wav")
            wet_list = glob.glob(f"{subset_dir}/validation_sequence/binaural.wav")
            pos_list = glob.glob(f"{subset_dir}/validation_sequence/tx_positions.txt")
        else:
            dry_list = glob.glob(f"{subset_dir}/subject*/mono.wav")
            wet_list = glob.glob(f"{subset_dir}/subject*/binaural.wav")
            pos_list = glob.glob(f"{subset_dir}/subject*/tx_positions.txt")

        n_data = len(dry_list)
        dry, wet, pos = [], [], []
        iterator = tqdm(range(n_data))
        iterator.set_description(f"[Loader] Load {mode} data")
        for i in iterator:
            dry.append(load_wav(dry_list[i]))    # 48k row : 1 sec
            wet.append(load_wav(wet_list[i]))    # 48k row : 1 sec
            pos.append(load_txt(pos_list[i]))    # 120 row : 1 sec
        dry = torch.cat(dry)    # (time,)
        wet = torch.cat(wet)    # (time, 2)
        pos = torch.cat(pos)    # (time, 7)

        self.wav_chunk = int(self.sr * lens_sec)
        self.txt_chunk = int(120 * lens_sec)
        self.n_chunk = dry.size(0) // self.wav_chunk

        # pad residuals
        wav_res = self.wav_chunk * (self.n_chunk+1) - dry.size(0)
        txt_res = self.txt_chunk * (self.n_chunk+1) - pos.size(0)
        dry = F.pad(dry, (0,wav_res))
        wet = F.pad(wet.transpose(0,1), (0,wav_res)).transpose(0,1)
        pos = F.pad(pos.transpose(0,1), (0,txt_res)).transpose(0,1)

        self.dry = rearrange(dry, '(b t c) -> b c t', t=self.wav_chunk, c=1)
        self.wet = rearrange(wet, '(b t) c -> b c t', t=self.wav_chunk)
        self.pos = rearrange(pos, '(b t) c -> b c t', t=self.txt_chunk)

    def __len__(self):
        return self.n_chunk

    def __getitem__(self, index):

        dry = self.dry.narrow(0,index,1).squeeze(0)  # (1, self.wav_chunk)
        wet = self.wet.narrow(0,index,1).squeeze(0)  # (2, self.wav_chunk)
        pos = self.pos.narrow(0,index,1).squeeze(0)  # (7, self.txt_chunk)

        return {
            "dry" : dry,
            "wet" : wet,
            "pos" : pos,
        }


class Trainset(GenericDataset):

    def __init__(
            self,
            mode='train',
            lens_sec=1.,
        ):
        super().__init__(
            mode=mode,
            lens_sec=lens_sec,
        )

class Testset(GenericDataset):

    def __init__(
            self,
            mode='valid',
            lens_sec=1.,
        ):
        super().__init__(
            mode=mode,
            lens_sec=lens_sec,
        )


if __name__=='__main__':
    dset = Trainset()
    _ = Testset(mode='valid')
    _ = Testset(mode='test')
    for i in range(10000):
        data = dset[i]
        print(data["dry"].shape, data["wet"].shape, data["pos"].shape)

