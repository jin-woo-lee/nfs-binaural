# Neural Fourier Shift for Binaural Speech Rendering
- This repository provides overall framework for training and evaluating Neural Fourier Shift (NFS) proposed in [Neural Fourier Shift for Binaural Speech Rendering](https://arxiv.org/abs/2211.00878)

### Prepare dataset

Download the [binaural speech dataset](https://github.com/facebookresearch/BinauralSpeechSynthesis/releases/tag/v1.0) and unzip it. See [WarpNet repo](https://github.com/facebookresearch/BinauralSpeechSynthesis#dataset) for more details about the data. When unzipped, create a symbolinc link to the directory that contains 'testset' and 'trainset' subdirectories as follows.

```bash
ls /path/to/binaural_dataset
#  testset  trainset
ln -s /path/to/binaural_dataset dataset/benchmark
```

### Training

Install third-party dependencies. This project was built and tested on RTX 2080 with CUDA 11.2.

```bash
pip install -r requirements.txt
```


### Training

To train NFS, run `main.py` with `--train` argument. You can also run several variants of NFS that appears in the ablation study by passing additional arguments as follows:

```bash
python main.py --gpus 0 --train                 # train NFS
python main.py --gpus 0 --train --wo_ni         # train wo.NI model
python main.py --gpus 0 --train --wo_lff        # train wo.LFF model
python main.py --gpus 0 --train --wo_geowarp    # train wo.GeoWarp model
python main.py --gpus 0 --train --wo_shifter    # train wo.Shifter model
```

### Inference

You can synthesize binaural audio by running `inference.py`. Specify the directory path where the mono `.wav` files and position `.txt` files for each are located.

```bash
python inference.py --gpu 0 --ckpt path/to/ckpt/file.pt --root_dir path/to/load/dir --save_dir path/to/save/dir
python inference.py --gpu 0 --ckpt results/nfs-00000000-000000/train/ckpt/0/nfs_1353.pt --root_dir dataset/benchmark/testset  --save_dir ./check --is_eval_set
```
