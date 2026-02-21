import torch
import torchaudio

from src.trainer.inferencer import Inferencer

class VocosInferencer(Inferencer):
    """
    Inferencer for Vocos vocoder inference.

    Extends Inferencer with:
    - Logging reconstructed audio and spectrograms
    - Custom process_batch for vocoder outputs
    """

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
            audio_hat = batch.get("audio_hat")
            if audio_hat is not None:
                sr = self.config.melspectrogram.sr
                for i in range(audio_hat.shape[0]):
                    audio_hat_i = audio_hat[i].detach().cpu()  # (T,)
                    torchaudio.save(
                        str(self.save_path / part / f"output_{batch_idx}_{i}.wav"),
                        audio_hat_i.unsqueeze(0),  # (1, T)
                        sample_rate=sr,
                    )

        return batch
    