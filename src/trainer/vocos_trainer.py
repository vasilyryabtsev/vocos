import torch

from src.logger.utils import plot_spectrogram
from src.loss.vocos_loss import DiscriminatorLoss, GeneratorLoss, FeatureMatchingLoss
from src.metrics.tracker import MetricTracker
from src.trainer.trainer import Trainer
from src.transforms import MelSpectrogram, MelSpectrogramConfig


class VocosTrainer(Trainer):
    """
    Trainer for the Vocos vocoder with GAN training.

    Extends Trainer with:
    - Discriminators (MPD + MRD) and their optimizer/scheduler
    - GAN training loop: alternating D and G updates per step
    - Composite generator loss (mel + adversarial + feature matching)
    - Checkpoint save/resume for discriminator state
    """

    def __init__(
        self,
        mel_config: MelSpectrogramConfig = MelSpectrogramConfig(),
        discriminator_mpd=None,
        discriminator_mrd=None,
        optimizer_d=None,
        lr_scheduler_d=None,
        **kwargs,
    ):
        self.mel_spec = MelSpectrogram(mel_config)
        self.discriminator_mpd = discriminator_mpd
        self.discriminator_mrd = discriminator_mrd
        self.optimizer_d = optimizer_d
        self.lr_scheduler_d = lr_scheduler_d

        super().__init__(**kwargs)

        self.disc_loss_fn = DiscriminatorLoss()
        self.gen_loss_fn = GeneratorLoss()
        self.feat_loss_fn = FeatureMatchingLoss()

        self.mel_loss_coeff = self.config.trainer.get("mel_loss_coeff", 45.0)
        self.feat_loss_coeff = self.config.trainer.get("feat_loss_coeff", 2.0)

    @property
    def _has_discriminator(self):
        return (
            self.discriminator_mpd is not None
            and self.discriminator_mrd is not None
        )

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        # Generator forward pass
        outputs = self.model(**batch)
        batch.update(outputs)

        audio_input = batch["audio_input"]
        audio_hat = batch["audio_hat"]

        if self.is_train and self._has_discriminator:
            # Discriminator step
            self._set_discriminator_requires_grad(True)
            self.optimizer_d.zero_grad()

            # MPD
            y_d_rs_mpd, y_d_gs_mpd, _, _ = self.discriminator_mpd(
                y=audio_input, y_hat=audio_hat.detach(),
            )
            loss_d_mpd, _, _ = self.disc_loss_fn(y_d_rs_mpd, y_d_gs_mpd)

            # MRD
            y_d_rs_mrd, y_d_gs_mrd, _, _ = self.discriminator_mrd(
                y=audio_input, y_hat=audio_hat.detach(),
            )
            loss_d_mrd, _, _ = self.disc_loss_fn(y_d_rs_mrd, y_d_gs_mrd)

            loss_d = loss_d_mpd + loss_d_mrd
            loss_d.backward()
            self.optimizer_d.step()

            # Generator step
            self._set_discriminator_requires_grad(False)
            self.optimizer.zero_grad()

            y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = (
                self.discriminator_mpd(y=audio_input, y_hat=audio_hat)
            )
            y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = (
                self.discriminator_mrd(y=audio_input, y_hat=audio_hat)
            )

            # Mel reconstruction loss
            mel_loss_dict = self.criterion(
                audio_hat=audio_hat, audio_input=audio_input,
            )
            mel_loss = mel_loss_dict["loss"]

            # Adversarial loss
            loss_g_mpd, _ = self.gen_loss_fn(y_d_gs_mpd)
            loss_g_mrd, _ = self.gen_loss_fn(y_d_gs_mrd)
            loss_g = loss_g_mpd + loss_g_mrd

            # Feature matching loss
            loss_feat_mpd = self.feat_loss_fn(fmap_rs_mpd, fmap_gs_mpd)
            loss_feat_mrd = self.feat_loss_fn(fmap_rs_mrd, fmap_gs_mrd)
            loss_feat = loss_feat_mpd + loss_feat_mrd

            # Total generator loss
            loss = (
                self.mel_loss_coeff * mel_loss
                + loss_g
                + self.feat_loss_coeff * loss_feat
            )

            loss.backward()
            self._clip_grad_norm()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            # Restore discriminator grad flag for next step
            self._set_discriminator_requires_grad(True)

            batch["loss"] = loss
            batch["mel_loss"] = mel_loss
            batch["gen_loss"] = loss_g
            batch["disc_loss"] = loss_d
            batch["feat_loss"] = loss_feat
        else:
            # Inference
            all_losses = self.criterion(**batch)
            batch.update(all_losses)
            if self._has_discriminator:
                batch.setdefault("mel_loss", batch["loss"])
                batch.setdefault("gen_loss", torch.tensor(0.0))
                batch.setdefault("disc_loss", torch.tensor(0.0))
                batch.setdefault("feat_loss", torch.tensor(0.0))

        # Update metrics
        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
                metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))

        return batch

    def _set_discriminator_requires_grad(self, requires_grad: bool):
        """Toggle discriminator gradient computation."""
        for p in self.discriminator_mpd.parameters():
            p.requires_grad = requires_grad
        for p in self.discriminator_mrd.parameters():
            p.requires_grad = requires_grad

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode != "train":
            self.log_spectrogram(**batch)
            self.log_audio(**batch)
    
    def log_spectrogram(self, audio_input, audio_hat, **batch):
        for i in range(audio_input.size(0)):
            audio_in = audio_input[i].detach().cpu()
            audio_out = audio_hat[i].detach().cpu()
            self._log_spectrogram(audio_in, name=f"input_spectrogram_{i}")
            self._log_spectrogram(audio_out, name=f"output_spectrogram_{i}")

    def _log_spectrogram(self, audio, name="spectrogram"):
        spec_to_plot = self.mel_spec(audio)
        image_array = plot_spectrogram(spec_to_plot.detach().cpu().numpy())
        self.writer.add_image(name, image_array, dataformats='HWC')


    def log_audio(self, audio_input, audio_hat, target_sr, **batch):
        """Log input and reconstructed audio."""
        for i in range(audio_input.size(0)):
            audio_in = audio_input[i].detach().cpu()
            audio_out = audio_hat[i].detach().cpu()
            if audio_in.dim() == 2:
                audio_in = audio_in.squeeze(0)
            if audio_out.dim() == 2:
                audio_out = audio_out.squeeze(0)
            self.writer.add_audio(
                f"audio_input_{i}", audio_in, sample_rate=target_sr[0],
            )
            self.writer.add_audio(
                f"audio_reconstructed_{i}", audio_out, sample_rate=target_sr[0],
            )

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        if self._has_discriminator:
            state["discriminator_mpd"] = self.discriminator_mpd.state_dict()
            state["discriminator_mrd"] = self.discriminator_mrd.state_dict()
            state["optimizer_d"] = self.optimizer_d.state_dict()
            if self.lr_scheduler_d is not None:
                state["lr_scheduler_d"] = self.lr_scheduler_d.state_dict()

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(
                    filename, str(self.checkpoint_dir.parent),
                )
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(
                    best_path, str(self.checkpoint_dir.parent),
                )
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file "
                "is different from that of the checkpoint. This may yield an "
                "exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (
            checkpoint["config"]["optimizer"] != self.config["optimizer"]
            or checkpoint["config"]["lr_scheduler"]
            != self.config["lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file "
                "is different from that of the checkpoint. Optimizer and "
                "scheduler parameters are not resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        if self._has_discriminator and "discriminator_mpd" in checkpoint:
            self.discriminator_mpd.load_state_dict(
                checkpoint["discriminator_mpd"],
            )
            self.discriminator_mrd.load_state_dict(
                checkpoint["discriminator_mrd"],
            )
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            if (
                self.lr_scheduler_d is not None
                and "lr_scheduler_d" in checkpoint
            ):
                self.lr_scheduler_d.load_state_dict(
                    checkpoint["lr_scheduler_d"],
                )

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )
