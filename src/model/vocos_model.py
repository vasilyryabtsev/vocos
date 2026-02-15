from torch import nn

from src.model.backbone import VocosBackbone
from src.model.heads import ISTFTHead


class VocosModel(nn.Module):
    """
    Vocos vocoder model.

    Consists of two components:
        1. Backbone with ConvNeXt that processes spectrogram features.
        2. ISTFTHead: ISTFT head that reconstructs the waveform.

    Args:
        backbone (VocosBackbone): Backbone network module.
        head (ISTFTHead): ISTFT head module for waveform reconstruction.
    """

    def __init__(self, backbone: VocosBackbone, head: ISTFTHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, spectrogram, audio, **batch):
        """
        Forward pass: spectrogram -> backbone -> waveform.

        Args:
            spectrogram (Tensor): Mel spectrogram of shape (B, n_mels, T).
            audio (Tensor): Original audio waveform of shape (B, 1, T).
        Returns:
            dict: output dict containing audio_hat.
        """
        x = self.backbone(spectrogram)
        audio_hat = self.head(x)

        audio_input = audio.squeeze(1)  # (B, 1, T) -> (B, T)
        if audio_hat.size(-1) > audio_input.size(-1):
            audio_hat = audio_hat[..., : audio_input.size(-1)]
        elif audio_hat.size(-1) < audio_input.size(-1):
            audio_input = audio_input[..., : audio_hat.size(-1)]

        return {
            "audio_hat": audio_hat,
            "audio_input": audio_input,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
