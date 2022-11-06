# https://github.com/facebookresearch/BinauralSpeechSynthesis/blob/main/LICENSE
"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
import numpy as np

class FourierTransform:
    def __init__(self,
                 fft_bins=2048,
                 win_length_ms=40,
                 frame_rate_hz=100,
                 causal=False,
                 preemphasis=0.0,
                 sample_rate=48000,
                 normalized=False):
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.preemphasis = preemphasis
        self.fft_bins = fft_bins
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate / frame_rate_hz)
        self.causal = causal
        self.normalized = normalized
        if self.win_length > self.fft_bins:
            print('FourierTransform Warning: fft_bins should be larger than win_length')

    def _convert_format(self, data, expected_dims):
        if not type(data) == torch.Tensor:
            data = torch.Tensor(data)
        if len(data.shape) < expected_dims:
            data = data.unsqueeze(0)
        if not len(data.shape) == expected_dims:
            raise Exception(f"FourierTransform: data needs to be a Tensor with {expected_dims} dimensions but got shape {data.shape}")
        return data

    def _preemphasis(self, audio):
        if self.preemphasis > 0:
            return torch.cat((audio[:, 0:1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1)
        return audio

    def _revert_preemphasis(self, audio):
        if self.preemphasis > 0:
            for i in range(1, audio.shape[1]):
                audio[:, i] = audio[:, i] + self.preemphasis * audio[:, i-1]
        return audio

    def _magphase(self, complex_stft):
        mag, phase = ta.functional.magphase(complex_stft, 1.0)
        return mag, phase

    def stft(self, audio):
        '''
        wrapper around torch.stft
        audio: wave signal as torch.Tensor
        '''
        hann = torch.hann_window(self.win_length)
        hann = hann.cuda() if audio.is_cuda else hann
        spec = torch.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
                       window=hann, center=not self.causal, normalized=self.normalized)
        return spec.contiguous()

    def complex_spectrogram(self, audio):
        '''
        audio: wave signal as torch.Tensor
        return: torch.Tensor of size channels x frequencies x time_steps (channels x y_axis x x_axis)
        '''
        self._convert_format(audio, expected_dims=2)
        audio = self._preemphasis(audio)
        return self.stft(audio)

    def magnitude_phase(self, audio):
        '''
        audio: wave signal as torch.Tensor
        return: tuple containing two torch.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        stft = self.complex_spectrogram(audio)
        return self._magphase(stft)

    def mag_spectrogram(self, audio):
        '''
        audio: wave signal as torch.Tensor
        return: magnitude spectrum as torch.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[0]

    def power_spectrogram(self, audio):
        '''
        audio: wave signal as torch.Tensor
        return: power spectrum as torch.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return torch.pow(self.mag_spectrogram(audio), 2.0)

    def phase_spectrogram(self, audio):
        '''
        audio: wave signal as torch.Tensor
        return: phase spectrum as torch.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[1]

    def mel_spectrogram(self, audio, n_mels):
        '''
        audio: wave signal as torch.Tensor
        n_mels: number of bins used for mel scale warping
        return: mel spectrogram as torch.Tensor of size channels x n_mels x time_steps for magnitude and phase spectrum
        '''
        spec = self.power_spectrogram(audio)
        mel_warping = ta.transforms.MelScale(n_mels, self.sample_rate)
        return mel_warping(spec)

    def complex_spec2wav(self, complex_spec, length):
        '''
        inverse stft
        complex_spec: complex spectrum as th.Tensor of size channels x frequencies x time_steps x 2 (real part/imaginary part)
        length: length of the audio to be reconstructed (in frames)
        '''
        complex_spec = self._convert_format(complex_spec, expected_dims=4)
        hann = torch.hann_window(self.win_length)
        hann = hann.cuda() if complex_spec.is_cuda else hann
        wav = ta.functional.istft(complex_spec, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length, window=hann, length=length, center=not self.causal)
        wav = self._revert_preemphasis(wav)
        return wav

    def magphase2wav(self, mag_spec, phase_spec, length):
        '''
        reconstruction of wav signal from magnitude and phase spectrum
        mag_spec: magnitude spectrum as torch.Tensor of size channels x frequencies x time_steps
        phase_spec: phase spectrum as torch.Tensor of size channels x frequencies x time_steps
        length: length of the audio to be reconstructed (in frames)
        '''
        mag_spec = self._convert_format(mag_spec, expected_dims=3)
        phase_spec = self._convert_format(phase_spec, expected_dims=3)
        complex_spec = torch.stack([mag_spec * torch.cos(phase_spec), mag_spec * torch.sin(phase_spec)], dim=-1)
        return self.complex_spec2wav(complex_spec, length)


class Loss(nn.Module):
    def __init__(self, mask_beginning=0):
        '''
        base class for losses that operate on the wave signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__()
        self.mask_beginning = mask_beginning

    def forward(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data = data[..., self.mask_beginning:]
        target = target[..., self.mask_beginning:]
        return self._loss(data, target)

    def _loss(self, data, target):
        pass


class L2Loss(Loss):
    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        return torch.mean((data - target).pow(2))


class AmplitudeLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data), self._transform(target)
        #------------------------------ 
        #data = torch.sum(data**2, dim=-1) ** 0.5
        #target = torch.sum(target**2, dim=-1) ** 0.5
        #return torch.mean(torch.abs(data - target))
        #------------------------------ 
        eps = torch.finfo(data.dtype).eps           # for numerical stability
        data = (data.pow(2).sum(-1)+eps).sqrt()
        target = (target.pow(2).sum(-1)+eps).sqrt()
        #------------------------------ 
        return torch.mean(torch.abs(data - target))

class PhaseLoss(Loss):
    def __init__(self, sample_rate, mask_beginning=0, ignore_below=0.1):
        '''
        :param sample_rate: (int) sample rate of the audio signal
        :param mask_beginning: (int) number of samples to mask at the beginning of the signal
        '''
        super().__init__(mask_beginning)
        self.ignore_below = ignore_below
        self.fft = FourierTransform(sample_rate=sample_rate)

    def _transform(self, data):
        return self.fft.stft(data.view(-1, data.shape[-1]))

    def _loss(self, data, target):
        '''
        :param data: predicted wave signals in a B x channels x T tensor
        :param target: target wave signals in a B x channels x T tensor
        :return: a scalar loss value
        '''
        data, target = self._transform(data).view(-1, 2), self._transform(target).view(-1, 2)
        # ignore low energy components for numerical stability
        target_energy = torch.sum(torch.abs(target), dim=-1)
        pred_energy = torch.sum(torch.abs(data.detach()), dim=-1)
        target_mask = target_energy > self.ignore_below * torch.mean(target_energy)
        pred_mask = pred_energy > self.ignore_below * torch.mean(target_energy)
        indices = torch.nonzero(target_mask * pred_mask).view(-1)
        data, target = torch.index_select(data, 0, indices), torch.index_select(target, 0, indices)
        # compute actual phase loss in angular space
        data_angles, target_angles = torch.atan2(data[:, 0], data[:, 1]), torch.atan2(target[:, 0], target[:, 1])
        loss = torch.abs(data_angles - target_angles)
        # positive + negative values in left part of coordinate system cause angles > pi
        # => 2pi -> 0, 3/4pi -> 1/2pi, ... (triangle function over [0, 2pi] with peak at pi)
        loss = np.pi - torch.abs(loss - np.pi)
        return torch.mean(loss)

def compute_metrics(binauralized, reference):
    '''
    compute l2 error, amplitude error, and angular phase error for the given binaural and reference singal
    :param binauralized: 2 x T tensor containing predicted binaural signal
    :param reference: 2 x T tensor containing reference binaural signal
    :return: errors as a scalar value for each metric and the number of samples in the sequence
    '''
    #binauralized, reference = binauralized.unsqueeze(0), reference.unsqueeze(0)

    # compute error metrics
    l2_error = L2Loss()(binauralized, reference)
    amplitude_error = AmplitudeLoss(sample_rate=48000)(binauralized, reference)
    phase_error = PhaseLoss(sample_rate=48000, ignore_below=0.2)(binauralized, reference)

    return{
        "l2": l2_error,
        "amplitude": amplitude_error,
        "phase": phase_error,
        "samples": binauralized.shape[-1]
    }


