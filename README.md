## SqueezeWave: High-Quality Efficient Neural Audio Synthesis
This is the implememtation of SqueezeWave. SqueezeWave achieved >25X speed up than WaveGlow without lossing performance. (modified 12/30/2019) 

The paper of SqueezeWave is here: TODO

More documents is here: https://docs.google.com/document/d/1VsgJ-Br-pGTGLIQXgPBWdavNR9yhHlj9OcvnUrJx79I/edit

Flop counts is here: https://colab.research.google.com/drive/1aV2u-u3fO2bTdtQVuMJMtA97PP5ht44E?usp=drive_open#scrollTo=BM3EBi5twBuz

## Setup

1. Clone our repo and initialize submodule

   ```command
   git clone https://github.com/BohanZhai/SqueezeWave.git
   cd SqueezeWave
   git submodule init
   git submodule update
   ```

2. Install requirements `pip3 install -r requirements.txt`

3. Install [Apex]


## Generate audio with our pre-existing model

1. Download our [published model]
2. Download [mel-spectrograms]
3. Generate audio `python3 inference.py -f <(ls mel_spectrograms/*.pt) -w waveglow_256channels.pt -o . --is_fp16 -s 0.6`  


## Train your own model

1. Download [LJ Speech Data]. In this example it's in `data/`

2. Make a list of the file names to use for training/testing

   ```command
   ls data/*.wav | tail -n+10 > train_files.txt
   ls data/*.wav | head -n10 > test_files.txt
   ```

3. Train your WaveGlow networks

   ```command
   mkdir checkpoints
   python train.py -c config.json
   ```

   For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.

   For mixed precision training set `"fp16_run": true` on `config.json`.

4. Make test set mel-spectrograms

   `python mel2samp.py -f test_files.txt -o . -c config.json`

5. Do inference with your network

   ```command
   ls *.pt > mel_files.txt
   python3 inference.py -f mel_files.txt -w checkpoints/SqueezeWave_10000 -o . --is_fp16 -s 0.6
   ```

[//]: # (TODO)
[//]: # (PROVIDE INSTRUCTIONS FOR DOWNLOADING LJS)
[pytorch 1.0]: https://github.com/pytorch/pytorch#installation
[website]: https://nv-adlr.github.io/WaveGlow
[paper]: https://arxiv.org/abs/1811.00002
[WaveNet implementation]: https://github.com/r9y9/wavenet_vocoder
[Glow]: https://blog.openai.com/glow/
[WaveNet]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/
[PyTorch]: http://pytorch.org
[published model]: https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing
[mel-spectrograms]: https://drive.google.com/file/d/1g_VXK2lpP9J25dQFhQwx7doWl_p20fXA/view?usp=sharing
[LJ Speech Data]: https://keithito.com/LJ-Speech-Dataset
[Apex]: https://github.com/nvidia/apex
