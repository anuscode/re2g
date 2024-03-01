from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    pretrained_model_name_or_path: str = "monologg/koelectra-base-v3-discriminator"

    learning_rate: float = 1e-5
    weight_decay: float = 1e-2

    context_max_length: int = 512
    context_padding: str = "longest"
    context_num_trainable_layers: int = 2

    question_max_length: int = 512
    question_padding: str = "longest"
    question_num_trainable_layers: int = 2

    datamodule_batch_size: int = 8
    dataloader_num_workers: int = 0

    trainer_limit_train_batches: int = 100
    trainer_max_epochs: int = 1
    trainer_limit_val_batches: int = 10
    trainer_strategy: str = "auto"  # deepspeed_stage_2
    trainer_precision: int = 32

    checkpoint_dirpath: str = "checkpoints"
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "max"


settings = Settings()
