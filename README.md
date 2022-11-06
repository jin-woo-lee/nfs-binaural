# Neural Fourier Shift for Binaural Speech Rendering
- This repository provides overall framework for training and evaluating Neural Fourier Shift (NFS) proposed in [Neural Fourier Shift for Binaural Speech Rendering](https://arxiv.org/abs/2211.00878)

### Prepare dataset

```bash
python3 dataset/collect_ood_data.py
ln -s /path/to/data/directory dataset/benchmark
ln -s /path/to/data/directory dataset/ood
```

### Training

training code. assume data directory is in ///

```bash
python3 main.py --gpus 0 --train                 # train NFS
python3 main.py --gpus 0 --train --wo_ni         # train wo.NI model
python3 main.py --gpus 0 --train --wo_lff        # train wo.LFF model
python3 main.py --gpus 0 --train --wo_geowarp    # train wo.GeoWarp model
python3 main.py --gpus 0 --train --wo_shifter    # train wo.Shifter model
```

### Ablation test

```bash
python3 main.py --gpus 0 --test --result_dir ... --load_epoch 16 --load_step 1353             
python3 main.py --gpus 0 --test --result_dir ... --load_epoch 16 --load_step 1353 --wo_ni     
python3 main.py --gpus 0 --test --result_dir ... --load_epoch 16 --load_step 1353 --wo_lff    
python3 main.py --gpus 0 --test --result_dir ... --load_epoch 16 --load_step 1353 --wo_geowarp
python3 main.py --gpus 0 --test --result_dir ... --load_epoch 16 --load_step 1353 --wo_shifter
```

### Inference

```bash
python3 inference.py --gpu 0 --ckpt path/to/ckpt/file.pt
```
