# Text-to-Speech (FastSpeech2) project

 This is a repository for Text-to-Speech project based on one of DLA course (HSE).
## Repository structure

`hw_3` - directory included all project files.
* `audio` - functions and classes for audio preprocessing based on NVIDIA.
* `base` - base classes for model, dataset and train.
* `best_results` - best audios during train process
* `collate_fn` - class preparing for dataloader and utils for padding.
* `configs` - configs with params for training.
* `datasets` - dataset class and text preprocessing functions with function for FastSpeechFirst.
* `logger` - files for logging.
* `loss` - definition for loss computation (both FastSpeechFirst and Second).
* `model` - architectures for both FastSpeechFirst and Second.
* `pitch_energy` - computation of pitch and energy and stats getting; functions for synthesis.
* `results_after_train` - this folder contains 81 examples of audio files with various configurations
* `trainer` - train loop, logging in W&B.
* `utils` - configs (dataclasses) with hyperparams of models (and for pitch&energy preparation) and other crucial functions (parse_config, object_loading, utils).

## Installation guide

Let's dive into several preparation steps you need to deal with before training FastSpeechSecond:

As usual, clone repository, change directory and install requirements:

```shell
!git clone https://github.com/KemmerEdition/HW-3-FS.git
!cd /content/HW-3-FS 
!pip install -r ./requirements.txt
```
## Train
You need to download data (all you need is run commands from `reproduce_train`), get pitch and energy.

Download pitch and energy files and unzip them.
```shell
!gdown --id 1Vx30j-pWYb2TxfU0UnqWIbREPeh_G9eb
!unzip pitch_energy_interpolation.npy.zip
```
Then train model with command below.
   ```shell
   !python -m train \
      -c hw_3/configs/fast_speech_second_first_try.json
   ```
## Test
You only need to run commands from `synthesis` and download checkpoint of my model, wait some time and enjoy.
   ```shell
   !gdown --id 1xKnOlTafYDP9p7BG6a7TXNsIMFwU1dnL
  ```
   ```shell
!python -m test \
   -c config.json \
   -r checkpoint-epoch350.pth 
   ```
Find directory named `results` and run following command, where you'll write path for audio you're interested in (see example):
```shell
from IPython import display
display.Audio('/content/HW-3-FS/results/pitch_s=0.8_0_waveglow.wav')
```
## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
