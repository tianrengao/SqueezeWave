## SqueezeWave: High-Quality Efficient Neural Audio Synthesis
By Tianren Gao *, Bohan Zhai *, Flora Xue, Daniel Rothchild, Bichen Wu, Joseph Gonzalez, and Kurt Keutzer (UC Berkeley)

We propose a new model called SqueezeWave which achieves 60x - 332x MAC reduction over WaveGlow without performance loss. 

Link to the paper: TODO

Audio samples of SqueezeWave are here: https://tianrengao.github.io/SqueezeWaveDemo/


We introduce 4 variants of SqueezeWave model in our paper. See the table below.


   | Model   | length | n_channels| MACs  | Reduction | MOS  |
   | ------- | ------ | --------- | ----- | --------- | ---- |
   |WaveGLow |  2048  | 8         | 228.9 | 1x        | TODO |
   |1        |  128   | 256       | 3.78  | 60x       | TODO |
   |2        |  64    | 256       | 2.16  | 106x      | TODO |
   |3        |  128   | 128       | 1.06  | 214x      | TODO |
   |4        |  64    | 128       | 0.68  | 332x      | TODO |


A detailed MAC calculation can be found from [here](https://colab.research.google.com/drive/13ZCmAMhHAcG6yixCofSMff2bp1om47mu)

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
   python train.py -c configs/config_a256_c128.json
   ```

   For multi-GPU training replace `train.py` with `distributed.py`.  Only tested with single node and NCCL.

   For mixed precision training set `"fp16_run": true` on `config.json`.

5. Make test set mel-spectrograms

   ```
   mkdir -p eval/mels
   python3 mel2samp.py -f test_files.txt -o eval/mels -c configs/config_a128_c256.json
   ```

6. Run inference on the test data. 

   ```command
   ls eval/mels > eval/mel_files.txt
   sed -i -e 's_.*_eval/mels/&_' eval/mel_files.txt
   mkdir -p eval/output
   python3 inference.py -f eval/mel_files.txt -w checkpoints/SqueezeWave_10000 -o eval/output --is_fp16 -s 0.6
   ```
## Credits
The implementation of this work is based on WaveGlow: https://github.com/NVIDIA/waveglow


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
