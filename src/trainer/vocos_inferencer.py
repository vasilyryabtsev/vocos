import PIL.Image
import torch
import torchaudio

from src.logger.utils import plot_spectrogram
from src.trainer.inferencer import Inferencer
from src.transforms import MelSpectrogram


class VocosInferencer(Inferencer):
    """
    Inferencer for Vocos vocoder inference.

    Extends Inferencer with:
    - Logging reconstructed audio and spectrograms
    - Custom process_batch for vocoder outputs
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_spec = MelSpectrogram(self.config.melspectrogram)

    def process_batch(self, batch_idx, batch, metrics, part):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        outputs = self.model(**batch)
        batch.update(outputs)

        self._log_batch(batch_idx, batch, mode="inference")

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        if self.save_path is not None:
            sr = self.config.melspectrogram.sr
            save_dir = self.save_path / part

            audio_dir = save_dir / "audio"
            spec_dir = save_dir / "spectrograms"
            audio_dir.mkdir(exist_ok=True, parents=True)
            spec_dir.mkdir(exist_ok=True, parents=True)

            audio_hat = batch["audio_hat"]
            audio_input = batch["audio_input"]
            spectrogram = batch.get("spectrogram")

            for i in range(audio_hat.shape[0]):
                stem = f"{batch_idx}_{i}"

                # Save original audio
                torchaudio.save(
                    str(audio_dir / f"orig_{stem}.wav"),
                    audio_input[i].detach().cpu().unsqueeze(0),
                    sample_rate=sr,
                )

                # Save generated audio
                torchaudio.save(
                    str(audio_dir / f"gen_{stem}.wav"),
                    audio_hat[i].detach().cpu().unsqueeze(0),
                    sample_rate=sr,
                )

                # Save original spectrogram
                if spectrogram is not None:
                    spec_np = spectrogram[i].detach().cpu().numpy()
                    img = plot_spectrogram(spec_np, name="original")
                    PIL.Image.fromarray(img).save(
                        str(spec_dir / f"orig_{stem}.png")
                    )

                # Save generated spectrogram
                mel_hat = self.mel_spec(
                    audio_hat[i].detach().cpu().unsqueeze(0)
                )[0]  # (n_mels, T')
                img_hat = plot_spectrogram(mel_hat.numpy(), name="generated")
                PIL.Image.fromarray(img_hat).save(
                    str(spec_dir / f"gen_{stem}.png")
                )

        return batch
