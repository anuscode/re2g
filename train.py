import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from re2g.configs import settings
from re2g.datasets.v1 import SquadV1DataModule
from re2g.models.dpr import DPR

PROJECT = "re2g"

MODEL_NAME = settings.pretrained_model_name_or_path

TRAINER_STRATEGY = settings.trainer_strategy

BATCH_SIZE = settings.datamodule_batch_size

CHECKPOINT_DIRPATH = settings.checkpoint_dirpath

CHECKPOINT_MONITOR = settings.checkpoint_monitor

CHECKPOINT_MODE = settings.checkpoint_mode


def main():

    dpr = DPR(MODEL_NAME)

    wandb.init(project=PROJECT, config=dpr.hparams)
    wandb.watch(dpr, log="all", log_freq=1)

    logger = WandbLogger(log_model="all")

    datamodule = SquadV1DataModule(
        pretrained_model_name_or_path=MODEL_NAME,
        batch_size=BATCH_SIZE,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=CHECKPOINT_DIRPATH,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MODE,
            every_n_train_steps=100,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=10,
        strategy=TRAINER_STRATEGY,
        logger=logger,
        callbacks=callbacks,
        limit_val_batches=10,
    )

    trainer.fit(model=dpr, datamodule=datamodule)


if __name__ == "__main__":
    main()
