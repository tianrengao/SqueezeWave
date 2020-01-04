## SqueezeWave: High-Quality Efficient Neural Audio Synthesis
We proposed a new model called SqueezeWave which achieves more than 25X speed up than SqueezeWave without lossing performance. This is the implememtation of SqueezeWave.  (modified 12/30/2019) 

The paper of SqueezeWave is here: TODO

The demo of SqueezeWave is here: TODO


We introduce 4 variants of SqueezeWave model in our paper. See the table below.

| Model  | L , C | Internal Channel | GFLOPs | MOS |
| ------------- | ------------- | ------------- | ------------- |------------- |
|1  | 128 , 128  | 256 -> 512  | 18 |TODO |
|2  | 64 , 256  | 256 -> 512  | 9 |TODO |
|3  | 128 , 128  | 128 -> 256  | TODO |TODO |
|4  | 64 , 256  | 128 -> 256  | TODO |TODO |

More details of these four models are here: https://docs.google.com/document/d/1VsgJ-Br-pGTGLIQXgPBWdavNR9yhHlj9OcvnUrJx79I/edit


Flop counts is here: https://colab.research.google.com/drive/1aV2u-u3fO2bTdtQVuMJMtA97PP5ht44E?usp=drive_open#scrollTo=BM3EBi5twBuz

## Setup
0. (Optional) Create a virtual environment

   ```
   virtualenv env
   source env/bin/activate
   ```

1. Clone our repo and initialize submodule

   ```command
   git clone https://github.com/BohanZhai/SqueezeWave.git
   cd SqueezeWave
   git submodule init
   git submodule update
   ```

2. Install requirements 
```pip3 install -r requirements.txt``` 

3. Install [Apex]
   ```1
   cd ../
   git clone https://www.github.com/nvidia/apex
   cd apex
   python setup.py install
   ```

## Generate audio with our pre-existing model

1. Download our [published model]
2. Download [mel-spectrograms]
3. (TODO) Generate audio `python3 inference.py -f <(ls mel_spectrograms/*.pt) -w SqueezeWave.pt -o . --is_fp16 -s 0.6`  


## Train your own model

1. Download [LJ Speech Data]. We assume all the waves are stored in the directory `^/data/`

2. Make a list of the file names to use for training/testing

   ```command
   ls data/*.wav | tail -n+10 > train_files.txt
   ls data/*.wav | head -n10 > test_files.txt
   ```

3. We provide 4 model configurations with audio channel and channel numbers specified in the table below. The configuration files are under ```^/configs``` directory. To choose the model you want to train, select the corresponding configuration file.

    | Model  | n_audio_channel | n_channels|
    | ------------- | ------------- | ------------- |
    |1  | 128  | 256 | 18 |
    |2  | 256  | 256  | 9 |
    |3  | 128  | 128 | TODO |
    |4  | 256  | 128 | TODO |

4. Train your SqueezeWave networks

   ```command
   mkdir checkpoints
   python train.py -c configs/config_g256_c128.json
   ```

   For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.

   For mixed precision training set `"fp16_run": true` on `config.json`.

5. Make test set mel-spectrograms

   `python mel2samp.py -f test_files.txt -o . -c config.json`

6. Do inference with your network

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
