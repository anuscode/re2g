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

TRAINER_PRECISION = settings.trainer_precision

TRAINER_LIMIT_TRAIN_BATCHES = settings.trainer_limit_train_batches

TRAINER_MAX_EPOCHS = settings.trainer_max_epochs

TRAINER_LIMIT_VAL_BATCHES = settings.trainer_limit_val_batches

DATAMODULE_BATCH_SIZE = settings.datamodule_batch_size

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
        batch_size=DATAMODULE_BATCH_SIZE,
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
        limit_train_batches=TRAINER_LIMIT_TRAIN_BATCHES,
        max_epochs=TRAINER_MAX_EPOCHS,
        limit_val_batches=TRAINER_LIMIT_VAL_BATCHES,
        strategy=TRAINER_STRATEGY,
        precision=TRAINER_PRECISION,
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model=dpr, datamodule=datamodule)


if __name__ == "__main__":
    main()
