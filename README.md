# RE2G

## 1. Installation

```bash
poetry install
```

## 2. Usage

#### Strongly recommend to create your own `.env` file and use the params as below.

```dotenv
# Optimizers
OPTIMIZER_LEARNING_RATE=1e-3
OPTIMIZER_WEIGHT_DECAY=1e-2

# Context Models
CONTEXT_MAX_LENGTH=512
CONTEXT_PADDING=longest
# The higher the number, the more trainable layers
CONTEXT_NUM_TRAINABLE_LAYERS=2

# Query Models
QUERY_MAX_LENGTH=512
QUERY_PADDING=longest
# The higher the number, the more trainable layers
QUERY_NUM_TRAINABLE_LAYERS=2

# Data module
# In this model, batch size is very important
DATAMODULE_BATCH_SIZE=128

# Dataloader
# If you are encountering issue on mac os,
# please set DATALOADER_NUM_WORKERS=0
DATALOADER_NUM_WORKERS=16

# Trainer
TRAINER_STRATEGY=auto
TRAINER_PRECISION=32
TRAINER_MAX_EPOCHS=100
TRAINER_LIMIT_VAL_BATCHES=1e-0
TRAINER_LIMIT_TRAIN_BATCHES=1e-0
TRAINER_LIMIT_TEST_BATCHES=1e-0

# Checkpoint
CHECKPOINT_EVERY_N_TRAIN_STEPS=0
CHECKPOINT_DIRPATH=checkpoints
CHECKPOINT_MONITOR=mrr
CHECKPOINT_MODE=max
CHECKPOINT_FOR_RESUME=
```

#### Log-in to wandb before training the model.

```bash
wandb login
```

#### Run the following command to train the model.

```bash
# which python
python train.py
```
- make sure to use the correct python version, if you have multiple python versions installed.
