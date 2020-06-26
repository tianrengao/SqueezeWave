import argparse
import json
import os
from scipy.io.wavfile import write
import torch

from mel2samp import files_to_list, MAX_WAV_VALUE
from denoiser import Denoiser
from parts.squeezewave import remove_weightnorm
from squeezewave import SqueezeWavePL


def main(model, mel_files, sigma, sampling_rate, denoiser_strength):
    mel_files = files_to_list(mel_files)

    if denoiser_strength > 0:
        denoiser = Denoiser(model)

    for i, file_path in enumerate(mel_files):
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        mel = torch.load(file_path)
        mel = torch.autograd.Variable(mel.to('cuda'))
        mel = torch.unsqueeze(mel, 0)

        with torch.no_grad():
            audio, _, _ = model.forward(mel, None, sigma=sigma)
            audio = audio.float()
            if denoiser_strength > 0:
                audio = denoiser(audio, denoiser_strength)
            audio = audio * MAX_WAV_VALUE

        audio = audio.squeeze()
        audio = audio.cpu().numpy()
        audio = audio.astype('int16')
        audio_path = os.path.join(
            os.getcwd(), '{}_synthesis.wav'.format(file_name))
        write(audio_path, sampling_rate, audio)
        print(audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='path to config file')
    parser.add_argument('-f', '--filelist_path', required=True,
                        help='list of test files')
    parser.add_argument('-w', '--squeezewave_path',
                        help='Path to squeezewave decoder checkpoint with model')
    parser.add_argument('-s', '--sigma', default=1.0, type=float)
    parser.add_argument('--sampling_rate', default=22050, type=int)
    parser.add_argument('-d', '--denoiser_strength', default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        data = f.read()
    config = json.loads(data)
    train_config = config['train_config']
    squeezewave_config = config['squeezewave_config']

    model = SqueezeWavePL(train_config['sigma'],
                          train_config['lr'],
                          **squeezewave_config).to('cuda').eval()
    checkpoint = torch.load(args.squeezewave_path)
    model.load_state_dict(checkpoint['state_dict'])

    main(
        model,
        args.filelist_path,
        args.sigma,
        args.sampling_rate,
        args.denoiser_strength,
    )
