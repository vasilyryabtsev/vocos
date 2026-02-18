import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import VocosTrainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="vocos_onebatchtest.yaml")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)

    metrics = {"train": [], "inference": []}
    for metric_type in ["train", "inference"]:
        for metric_config in config.metrics.get(metric_type, []):
            metrics[metric_type].append(instantiate(metric_config))

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = None
    if lr_scheduler in config:
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    # discriminators
    discriminator_mpd = instantiate(config.discriminator_mpd).to(device)
    discriminator_mrd = instantiate(config.discriminator_mrd).to(device)
    logger.info(f"MPD params: {sum(p.numel() for p in discriminator_mpd.parameters()):,}")
    logger.info(f"MRD params: {sum(p.numel() for p in discriminator_mrd.parameters()):,}")

    d_params = list(discriminator_mpd.parameters()) + list(discriminator_mrd.parameters())
    optimizer_d = instantiate(config.optimizer_d, params=d_params)
    lr_scheduler_d = None
    if lr_scheduler_d in config:
        lr_scheduler_d = instantiate(config.lr_scheduler_d, optimizer=optimizer_d)

    # base trainer kwargs
    trainer_kwargs = dict(
        model=model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
        discriminator_mpd=discriminator_mpd,
        discriminator_mrd=discriminator_mrd,
        optimizer_d=optimizer_d,
        lr_scheduler_d=lr_scheduler_d,
    )

    trainer = VocosTrainer(**trainer_kwargs)
    trainer.train()


if __name__ == "__main__":
    main()
