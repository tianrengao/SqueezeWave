import argparse
import json
import os
import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from squeezewave import SqueezeWavePL, SqueezeWaveLoss
from mel2samp import Mel2Samp


def main(train_config, data_config, squeezewave_config, num_gpus):
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])

    # model
    model = SqueezeWavePL(train_config['sigma'],
                          train_config['lr'],
                          **squeezewave_config)

    # dataloader
    n_group = squeezewave_config['n_group']
    train_dataset = Mel2Samp(n_group, **data_config)
    train_dataloader = DataLoader(train_dataset,
                                  num_workers=4,
                                  shuffle=True,
                                  batch_size=train_config['batch_size'],
                                  pin_memory=False,
                                  drop_last=True)

    # trainer
    checkpoint_callback = ModelCheckpoint(filepath=train_config['output_directory'],
                                          monitor='loss',
                                          save_weights_only=False,
                                          save_last=True)
    trainer = Trainer(max_epochs=train_config['epochs'],
                      gpus=num_gpus,
                      amp_level='O1',
                      use_amp=train_config['fp16_run'],
                      distributed_backend='ddp',
                      checkpoint_callback=checkpoint_callback,
                      benchmark=False)

    trainer.fit(model, train_dataloader=train_dataloader)


if __name__ == '__main__':
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='JSON file for configuration')
    parser.add_argument('-g', '--gpus', type=int, default=1,
                        help='number of gpus to train on')
    args = parser.parse_args()

    # load configs
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config['train_config']
    data_config = config['data_config']
    squeezewave_config = config['squeezewave_config']

    # configure gpus
    num_gpus = args.gpus
    if num_gpus > torch.cuda.device_count():
        raise Exception('Only {} gpus available ({} specified)'
                        .format(torch.cuda.device_count(), num_gpus))
