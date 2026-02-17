import torch
import torchaudio

from torch import nn
from typing import List, Tuple

from src.transforms import MelSpectrogramConfig, MelSpectrogram
from src.model.utils import safe_log


class MelSpecReconstructionLoss(nn.Module):
    """
    L1 distance between the mel-scaled magnitude spectrograms of the ground truth
    sample and the generated sample.

    Args:
        sample_rate (int): Sample rate of the audio.
        n_fft (int): FFT size.
        hop_length (int): Hop length.
        n_mels (int): Number of mel bins.
    """

    def __init__(
        self, config: MelSpectrogramConfig = MelSpectrogramConfig(),
    ):
        super().__init__()
        self.mel_spec = MelSpectrogram(config=config)

    def forward(self, audio_hat, audio_input, **batch):
        """
        Args:
            audio_hat (Tensor): Predicted audio waveform (B, T).
            audio_input (Tensor): Ground truth audio waveform (B, T).

        Returns:
            dict: Dictionary with "loss" key.
        """
        mel_hat = safe_log(self.mel_spec(audio_hat))
        mel = safe_log(self.mel_spec(audio_input))
        loss = torch.nn.functional.l1_loss(mel, mel_hat)
        return {"loss": loss}


class GeneratorLoss(nn.Module):
    """
    Generator Loss module. 
    Calculates the loss for the generator based on discriminator outputs.
    """

    def forward(self, disc_outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_outputs[0].device, dtype=disc_outputs[0].dtype)
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean(torch.clamp(1 - dg, min=0))
            gen_losses.append(l)
            loss += l
        return loss, gen_losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator Loss module. Calculates the loss for the discriminator based on real and generated outputs.
    """

    def forward(
        self, disc_real_outputs: List[torch.Tensor], disc_generated_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        loss = torch.zeros(1, device=disc_real_outputs[0].device, dtype=disc_real_outputs[0].dtype)
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean(torch.clamp(1 - dr, min=0))
            g_loss = torch.mean(torch.clamp(1 + dg, min=0))
            loss += r_loss + g_loss
            r_losses.append(r_loss)
            g_losses.append(g_loss)
        return loss, r_losses, g_losses


class FeatureMatchingLoss(nn.Module):
    """
    Feature Matching Loss module. Calculates the feature matching loss between
    feature maps of the sub-discriminators.
    """

    def forward(self, fmap_r: List[List[torch.Tensor]], fmap_g: List[List[torch.Tensor]]) -> torch.Tensor:
        loss = torch.zeros(1, device=fmap_r[0][0].device, dtype=fmap_r[0][0].dtype)
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss
