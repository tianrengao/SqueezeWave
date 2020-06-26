import torch
from torch.optim import Adam
import librosa
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from parts import SqueezeWave, remove_weightnorm


class SqueezeWavePL(LightningModule):
    '''
    SqueezeWavePL implements the SqueezWave model in whole. This PL is meant to
    be used during training
    Args:
        sigma (float): Standard deviation of the normal distribution from which
            we sample z. Defaults to 1.0.
        lr (float): Learning rate of Adam optimizer.
            Defaults to 4e-4.
        n_mel_channels (int): Size of input mel spectrogram
            Defaults to 80.
        n_flows (int): Number of normalizing flows/layers of squeezewave.
            Defaults to 12
        n_group (int): Each audio/spec pair is split in n_group number of
            groups. It must be divisible by 2 as halves are split this way.
            Defaults to 256
        n_early_every (int): After n_early_every layers, n_early_size number of
            groups are skipped to the output of the Neural Module.
            Defaults to 4
        n_early_size (int): The number of groups to skip to the output at every
            n_early_every layers.
            Defaults to 2
        n_wn_layers (int): The number of layers of the wavenet submodule.
            Defaults to 8
        n_wn_channels (int): The number of channels of the wavenet submodule.
            Defaults to 128
        wn_kernel_size (int): The kernel size of the wavenet submodule.
            Defaults to 3
    '''
    def __init__(
        self,
        sigma: float = 1.0,
        lr: float = 4e-4,
        n_mel_channels: int = 80,
        n_flows: int = 12,
        n_group: int = 256,
        n_early_every: int = 4,
        n_early_size: int = 2,
        n_wn_layers: int = 128,
        n_wn_channels: int = 128,
        wn_kernel_size: int = 3,
    ):
        super().__init__()
        self._lr = lr
        WN_config = {
            'n_layers': n_wn_layers,
            'n_channels': n_wn_channels,
            'kernel_size': wn_kernel_size,
        }
        self.squeezewave = SqueezeWave(
            n_mel_channels=n_mel_channels,
            n_flows=n_flows,
            n_group=n_group,
            n_early_every=n_early_every,
            n_early_size=n_early_size,
            WN_config=WN_config,
        )
        self.loss_fn = SqueezeWaveLoss(sigma)

    def forward(self, mel_spectrogram, audio, sigma=1.0):
        '''
        mel_spectrogram:  batch x n_mel_channels x frames
        audio: batch x time
        '''
        if self.training:
            audio, log_s_list, log_det_W_list = self.squeezewave.forward((mel_spectrogram, audio))
        else:
            audio = self.squeezewave.infer(mel_spectrogram, sigma=sigma)
            log_s_list = log_det_W_list = []

        return audio, log_s_list, log_det_W_list

    def training_step(self, batch, batch_idx):
        mel_spectrogram, audio = batch

        # pytorch lightning default puts tensors on gpu/cpu
        mel_spectrogram = torch.autograd.Variable(mel_spectrogram)
        audio = torch.autograd.Variable(audio)
        out = self.forward(mel_spectrogram, audio)

        # return outputs and compute loss later (for distributed training)
        return {'out': out}

    def training_step_end(self, outputs):
        loss = self.loss_fn(outputs['out'])
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = Adam([p for p in self.squeezewave.parameters() if p.requires_grad], lr=self._lr)
        return [optimizer]

    def configure_apex(self, amp, model, optimizers, amp_level):
        model, optimizer = amp.initialize(model, optimizers[0], opt_level=amp_level)
        return model, [optimizer]


class SqueezeWaveLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(SqueezeWaveLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]

        loss = torch.sum(z * z)/(2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))
