#!/usr/bin/env python3
import argparse
import os
import random
import csv
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchaudio import datasets
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from utils import *
from torch.autograd import Variable
import torch.optim as optim
import torchaudio.functional as TAF
import torchaudio.transforms as TAT
from tqdm import tqdm
import wandb
from evaluate import compute_metrics
from auraloss.freq import MultiResolutionSTFTLoss as MRSTFT
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ

class Solver(object):
    def __init__(self, args):
        self.mp_context = torch.multiprocessing.get_context('fork')
        self.train_loss = {}
        self.valid_loss = {}
        self.sr    = args.sr

    def set_dataset(self, args):
        module = __import__('dataset.loader', fromlist=[''])
        if args.train:
            self.trainset = module.Trainset(lens_sec=args.lens_sec)
            self.validset = module.Testset(mode='valid',lens_sec=args.test_lens_sec)
        if args.test:
            self.testset = module.Testset(mode='test',lens_sec=args.test_lens_sec)

    def set_gpu(self, args):
        print('set distributed data parallel')
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        #------------------------------ 
        # define data loaders 
        #------------------------------ 
        if args.train:
            self.train_sampler = DistributedSampler(self.trainset,shuffle=True,rank=args.gpu,seed=args.seed)
            self.valid_sampler = DistributedSampler(self.validset,shuffle=False,rank=args.gpu)
            self.train_loader = DataLoader(
                self.trainset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.train_sampler, drop_last=True,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )
            self.valid_loader = DataLoader(
                self.validset, batch_size=args.test_batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.valid_sampler, drop_last=False,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )
        if args.test:
            self.test_sampler = DistributedSampler(self.testset,shuffle=False,rank=args.gpu)
            self.test_loader = DataLoader(
                self.testset, batch_size=args.test_batch_size,
                shuffle=False, num_workers=args.num_workers,
                multiprocessing_context=self.mp_context,
                pin_memory=True, sampler=self.test_sampler, drop_last=False,
                worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed())%(2**32-1)),
            )
    
        self.n_fft = args.n_fft
        self.win_l = int(1e-3 * args.stft_window_ms * self.sr)
        self.hop_l = int(1e-3 * args.stft_hoplen_ms * self.sr)
        #------------------------------ 
        # define models 
        #------------------------------ 
        if args.train:
            gen = __import__('networks.nfs', fromlist=[''])
        if args.test:
            code_path = f"results.{args.result_dir}.codes"
            gen = __import__(f'{code_path}.networks.nfs', fromlist=[''])
        nfs = gen.NFS(
            window_ms=args.model_window_ms, nch=args.channel, cdim=args.cdim,
            wo_ni=args.wo_ni, wo_lff=args.wo_lff, wo_geowarp=args.wo_geowarp, wo_shifter=args.wo_shifter,
        )

        if args.verbose and args.gpu == 0:
            print(nfs)
   
        params = list(nfs.parameters())
        opt_conf = { 'lr': args.lr, 'betas': (0.9, 0.999), }
        sch_conf = { 'step_size': 1, 'gamma': 0.9, }
        self.optim = get_optimizer(args.optimizer, params, opt_conf)
        self.sched = get_scheduler(args.scheduler, self.optim, sch_conf)
        n_params = sum([param.view(-1).size()[0] for param in nfs.parameters()])
        print(f"num. params: {n_params}")

        torch.cuda.set_device(args.gpu)

        # Distribute models to machine
        nfs = nfs.to('cuda:{}'.format(args.gpu))
        ddp_nfs = torch.nn.parallel.DistributedDataParallel(
            nfs,
            device_ids=[args.gpu],
            output_device=args.gpu,
            #find_unused_parameters=True,
            broadcast_buffers=False,    # for batchnorm
        )
        self.nfs = ddp_nfs

        if args.resume or args.test:
            print("Load checkpoint from: {}".format(args.ckpt))
            checkpoint = torch.load(args.ckpt, map_location=f'cuda:{args.gpu}')
            self.nfs.load_state_dict(checkpoint["nfs"])
            self.start_epoch = (int)(checkpoint["epoch"])
            self.epoch = (int)(checkpoint["epoch"])
            self.optim.load_state_dict(checkpoint["optim"])
        else:
            self.start_epoch = 0

        #------------------------------ 
        # define other modules 
        #------------------------------ 
        n_mels = args.n_mel; self.n_freq = self.n_fft // 2 + 1
        self.linspec = TAT.Spectrogram(
            n_fft = self.n_fft, win_length = self.win_l, hop_length = self.hop_l,
            power = None,
        ).to(f"cuda:{args.gpu}")
        self.melspec = TAT.MelSpectrogram(
            sample_rate=self.sr,
            n_fft = self.n_fft, win_length = self.win_l, hop_length = self.hop_l,
            n_mels = n_mels, f_min = 0, f_max=self.sr // 2,
        ).to(f"cuda:{args.gpu}")
        basis = mel_basis(self.sr, self.n_fft, n_mels).narrow(1,0,self.n_freq-1)
        omega = torch.linspace(0, - self.sr, self.n_freq).view(1,1,-1,1).narrow(2,1,self.n_freq-1)

        mrstft = MRSTFT()
        wbpesq = PESQ(16000, 'wb')

        self.basis = basis.to(f"cuda:{args.gpu}")
        self.omega = omega.to(f"cuda:{args.gpu}")
        self.mrstft = mrstft.to(f"cuda:{args.gpu}")
        self.wbpesq = wbpesq.to(f"cuda:{args.gpu}")

    def save_checkpoint(self, args, epoch, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "nfs": self.nfs.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch,
            "step" : step}
        checkpoint_path = os.path.join(checkpoint_dir,'{}_{}.pt'.format(args.model, step))
        torch.save(checkpoint_state, checkpoint_path)
        print("Saved checkpoint: {}".format(checkpoint_path))
   
    def fourier(self, x, method='stft'):
        eps = torch.finfo(x.dtype).eps
        spec = self.linspec(x)
        mag = spec.abs() + eps     # (B, C, F, T)
        mel = self.melspec(x) + eps

        if mag.size(1) == 2:
            ll, rl = mag.log10().mean(2).chunk(2, 1)
            dbild = ll.squeeze(1) - rl.squeeze(1)
        else:
            dbild = None
        phs = torch.view_as_real(spec / mag)
        return {
            "logmag"   : 20 * mag.log10(),
            "logmel"   : 20 * mel.log10(),
            "dbild"    : dbild,
        }

    def train(self, args):
    
        self.nfs.train()
   
        root_dir = f'results/{args.result_dir}/train'
        for epoch in range(self.start_epoch, args.total_epochs+1):
            self.epoch = epoch
            iterator = tqdm(self.train_loader) if args.gpu==0 else self.train_loader
            for i, ts in enumerate(iterator):
                self.step = self.epoch*len(self.train_loader) + i

                dry = ts["dry"].cuda(args.gpu,non_blocking=True).float()  # (B,1,W)
                wet = ts["wet"].cuda(args.gpu,non_blocking=True).float()  # (B,2,W)
                pos = ts["pos"].cuda(args.gpu,non_blocking=True).float()  # (B,7,T)

                #============================== 
                """ Train NFS """
                #============================== 
                est, enc, _, ang = self.nfs(pos, dry)

                dryf = self.fourier(dry, 'stft')
                wetf = self.fourier(wet, 'stft')
                estf = self.fourier(est, 'stft')
                dry_mag, dry_mel = dryf["logmag"], dryf["logmel"]
                wet_mag, wet_mel = wetf["logmag"], wetf["logmel"]
                est_mag, est_mel = estf["logmag"], estf["logmel"]
                est_mag, est_mel = estf["logmag"], estf["logmel"]
                est_ild = estf["dbild"]
                wet_ild = wetf["dbild"]
                estl, estr = est.chunk(2,1); wetl, wetr = wet.chunk(2,1)
                est_lrms = torch.cat((est, estl+estr, estl-estr), dim=1)
                wet_lrms = torch.cat((wet, wetl+wetr, wetl-wetr), dim=1)

                wet_mag_loss = args.tf_c * spec_loss(est_mag, wet_mag)
                wet_mel_loss = args.tf_c * spec_loss(est_mel, wet_mel)
                est_mag_left, est_mag_right = est_mag.chunk(2, 1)
                est_mel_left, est_mel_right = est_mel.chunk(2, 1)
                wet_mag_left, wet_mag_right = wet_mag.chunk(2, 1)
                wet_mel_left, wet_mel_right = wet_mel.chunk(2, 1)
                est_mag_diff = est_mag_left - est_mag_right
                est_mel_diff = est_mel_left - est_mel_right
                wet_mag_diff = wet_mag_left - wet_mag_right
                wet_mel_diff = wet_mel_left - wet_mel_right
                wet_ild_loss = args.tf_c * spec_loss(est_mag_diff, wet_mag_diff) \
                             + args.tf_c * spec_loss(est_mel_diff, wet_mel_diff)
                scr = compute_metrics(est, wet)
                l_2, amp, phs = 1e3 * scr["l2"], scr["amplitude"], scr["phase"]

                snr = SI_SNR(est, wet)
                sdr = SI_SDR(est, wet)
                mrstft_loss = self.mrstft(est, wet)

                loss = []
                loss += [l_2] if args.l2 else [0]
                loss += [phs] if args.phs else [0]
                loss += [mrstft_loss] if args.mrstft else [0]
                loss += [wet_ild_loss] if args.ild else [0]

                loss = [l for l in loss if l != 0]
                loss = sum(loss) / len(loss)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if args.gpu==0:
                    if i % args.plot_iter == 0:
                        dry_mag = dry_mag.detach().cpu().numpy()
                        wet_mag = wet_mag.detach().cpu().numpy()
                        est_mag = est_mag.detach().cpu().numpy()
                        dry_mel = dry_mel.detach().cpu().numpy()
                        wet_mel = wet_mel.detach().cpu().numpy()
                        est_mel = est_mel.detach().cpu().numpy()
                        dry_np = dry.detach().cpu().numpy()
                        wet_np = wet.detach().cpu().numpy()
                        est_np = est.detach().cpu().numpy()
                        enc_np = enc.detach().cpu().numpy()
                        wet_ild = wet_ild.detach().cpu().numpy()
                        est_ild = est_ild.detach().cpu().numpy()

                        plot_dir = f"{root_dir}/plot/{epoch}"
                        os.makedirs(plot_dir, exist_ok=True)
                        plot_b = np.random.randint(dry_np.shape[0])
                        plots, wavs = save_samples(
                            [[dry_mag, wet_mag, est_mag],
                             [dry_mel, wet_mel, est_mel],
                             [dry_np, wet_np, est_np],
                             [wet_ild, est_ild, enc_np],
                            ],
                            plot_dir, i,
                            self.sr,
                            b = plot_b,
                        )
    
                    if i % 10 == 0:
                        tot = loss
                        scr = compute_metrics(est, wet)
                        l_2, amp, phs = 1e3 * scr["l2"], scr["amplitude"], scr["phase"]

                        tot = tot.item()
                        wet_mag = wet_mag_loss.item()
                        wet_mel = wet_mel_loss.item()
                        snr = snr.item(); sdr = sdr.item()
                        l_2, amp, phs = l_2.item(), amp.item(), phs.item()
                        wet_ild = wet_ild_loss.item()
                        mrstft = mrstft_loss.item()

                        self.train_loss['train-total'] = tot
                        self.train_loss['train-wet-logmag'] = wet_mag
                        self.train_loss['train-wet-logmel'] = wet_mel
                        self.train_loss['train-wet-ild'] = wet_ild
                        self.train_loss['train-snr'] = snr
                        self.train_loss['train-sdr'] = sdr
                        self.train_loss['train-l_2'] = l_2
                        self.train_loss['train-amp'] = amp
                        self.train_loss['train-phs'] = phs
                        self.train_loss['train-mrstft'] = mrstft

                        lr = self.optim.param_groups[0]['lr']
                        iterator.set_postfix(
                            TOTAL = tot,
                            wet_mag = wet_mag,
                            wet_mel = wet_mel,
                            wet_ild = wet_ild,
                            l_2=l_2, amp=amp, phs=phs,
                            mrstft = mrstft,
                            lr = lr,
                        )
    
            if epoch % args.save_epoch == 0:
                checkpoint_dir = f'{root_dir}/ckpt/{epoch}'
                self.save_checkpoint(args, epoch, i, checkpoint_dir)
            if epoch % args.valid_epoch == 0:
                self.test(args, 'valid')

            self.sched.step()
            self.nfs.train()
        # end of training
    
    def test(self, args, mode='test'):
    
        self.nfs.eval()

        SNR_EVAL = []; SDR_EVAL = []
        TOT_LOSS = []; DLY_LOSS = []; LGT_LOSS = []
        WET_MAG_LOSS = []; WET_MEL_LOSS = []; WET_SPC_LOSS = []
        WET_GDM_LIN_LOSS = []; WET_GDS_LIN_LOSS = []
        WET_GDM_MEL_LOSS = []; WET_GDS_MEL_LOSS = []
        WET_CLS_LOSS = []; EST_CLS_LOSS = []; EST_EMB_LOSS = []
        WET_ILD_LOSS = []
        L_2_EVAL = []; AMP_EVAL = []; PHS_EVAL = []
        MRSTFT_EVAL = [];  WBPESQ_EVAL = []
        num_samples = 0

        root_dir = f'results/{args.result_dir}/{mode}'
        os.makedirs(root_dir, exist_ok=True)

        loader = self.test_loader if mode.lower() == 'test' else self.valid_loader
        iterator = tqdm(loader) if args.gpu==0 else loader
        for i, ts in enumerate(iterator):
            dry = ts["dry"].cuda(args.gpu,non_blocking=True).float()  # (B,1,W)
            wet = ts["wet"].cuda(args.gpu,non_blocking=True).float()  # (B,2,W)
            pos = ts["pos"].cuda(args.gpu,non_blocking=True).float()  # (B,7,T)

            with torch.no_grad():
                est, enc, _, _ = self.nfs(pos, dry)

            dryf = self.fourier(dry, 'stft')
            wetf = self.fourier(wet, 'stft')
            estf = self.fourier(est, 'stft')
            dry_mag, dry_mel = dryf["logmag"], dryf["logmel"]
            wet_mag, wet_mel = wetf["logmag"], wetf["logmel"]
            est_mag, est_mel = estf["logmag"], estf["logmel"]
            est_ild = estf["dbild"]
            wet_ild = wetf["dbild"]

            wet_mag_loss = args.tf_c * spec_loss(est_mag, wet_mag)
            wet_mel_loss = args.tf_c * spec_loss(est_mel, wet_mel)
            wet_ild_loss = args.ld_c * F.mse_loss(est_ild, wet_ild)

            loss = wet_mag_loss + wet_mel_loss \
                 + wet_ild_loss

            ere = TAF.resample(est, self.sr, 16000, resampling_method='kaiser_window')
            wre = TAF.resample(wet, self.sr, 16000, resampling_method='kaiser_window')

            snr = SI_SNR(est, wet)
            sdr = SI_SDR(est, wet)
            scr = compute_metrics(est, wet)
            l_2, amp, phs = 1e3 * scr["l2"], scr["amplitude"], scr["phase"]
            mrstft_loss = self.mrstft(est, wet)
            wbpesq_loss = self.wbpesq(ere, wre)

            batch_size = dry.size(0)
            TOT_LOSS.append(loss.item() * batch_size)
            WET_MAG_LOSS.append(wet_mag_loss.item() * batch_size)
            WET_MEL_LOSS.append(wet_mel_loss.item() * batch_size)
            WET_ILD_LOSS.append(wet_ild_loss.item() * batch_size)
            SNR_EVAL.append(snr.item() * batch_size)
            SDR_EVAL.append(sdr.item() * batch_size)
            L_2_EVAL.append(l_2.item() * batch_size)
            AMP_EVAL.append(amp.item() * batch_size)
            PHS_EVAL.append(phs.item() * batch_size)
            MRSTFT_EVAL.append(mrstft_loss.item() * batch_size)
            WBPESQ_EVAL.append(wbpesq_loss.item() * batch_size)
            num_samples += batch_size

            if args.gpu==0 and not args.dont_plot:
                dry_mag = dry_mag.detach().cpu().numpy()
                wet_mag = wet_mag.detach().cpu().numpy()
                est_mag = est_mag.detach().cpu().numpy()
                dry_mel = dry_mel.detach().cpu().numpy()
                wet_mel = wet_mel.detach().cpu().numpy()
                est_mel = est_mel.detach().cpu().numpy()
                dry = dry.detach().cpu().numpy()
                wet = wet.detach().cpu().numpy()
                est = est.detach().cpu().numpy()
                enc = enc.detach().cpu().numpy()
                wet_ild = wet_ild.detach().cpu().numpy()
                est_ild = est_ild.detach().cpu().numpy()

                plot_dir = f"{root_dir}/plot/{self.epoch}"
                os.makedirs(plot_dir, exist_ok=True)
                plot_b = np.random.randint(dry.shape[0])
                plots, wavs = save_samples(
                    [[dry_mag, wet_mag, est_mag],
                     [dry_mel, wet_mel, est_mel],
                     [dry, wet, est],
                     [wet_ild, est_ild, enc],
                    ],
                    plot_dir, i,
                    self.sr,
                    b = plot_b,
                )
            # end of test iter
        tot_loss = sum(TOT_LOSS) / num_samples
        wet_mag_loss = sum(WET_MAG_LOSS) / num_samples
        wet_mel_loss = sum(WET_MEL_LOSS) / num_samples
        wet_ild_loss = sum(WET_ILD_LOSS) / num_samples
        snr = sum(SNR_EVAL) / num_samples
        sdr = sum(SDR_EVAL) / num_samples
        l_2 = sum(L_2_EVAL) / num_samples
        amp = sum(AMP_EVAL) / num_samples
        phs = sum(PHS_EVAL) / num_samples
        mrstft = sum(MRSTFT_EVAL) / num_samples
        wbpesq = sum(WBPESQ_EVAL) / num_samples

        if mode.lower() == 'valid':
            self.valid_loss['valid-total'] = tot_loss
            self.valid_loss['valid-wet-logmag'] = wet_mag_loss
            self.valid_loss['valid-wet-logmel'] = wet_mel_loss
            self.valid_loss['valid-wet-ild'] = wet_ild_loss
            self.valid_loss['valid-snr'] = snr
            self.valid_loss['valid-sdr'] = sdr
            self.valid_loss['valid-l_2'] = l_2
            self.valid_loss['valid-amp'] = amp
            self.valid_loss['valid-phs'] = phs
            self.valid_loss['valid-mrstft'] = mrstft
            self.valid_loss['valid-pesq'] = wbpesq
        else:
            LOG = {
                "l_2"   : l_2,
                "amp"   : amp,
                "phs"   : phs,
                "pesq"  : wbpesq,
                "mrstft": mrstft,
                "SNR"   : snr,
                "SDR"   : sdr,
            }
            print(LOG)
            log_path = f"{root_dir}/result/ep-{self.epoch}.txt"
            os.makedirs(f"{root_dir}/result", exist_ok=True)
            with open(log_path, 'w') as f:
                for key in LOG.keys():
                    f.write(f"{key}\t{LOG[key]}\n")
        # end of test
   

