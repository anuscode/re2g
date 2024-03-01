import lightning as L

from re2g.datasets.v1 import SquadV1DataModule
from re2g.models.dpr import DPR

dpr = DPR(
    pretrained_model_name_or_path="monologg/koelectra-base-v3-discriminator",
)

squad_v1_data_module = SquadV1DataModule(
    pretrained_model_name_or_path="monologg/koelectra-base-v3-discriminator",
    batch_size=64,
)

trainer = L.Trainer(limit_train_batches=100, max_epochs=100000)
trainer.fit(model=dpr, train_dataloaders=squad_v1_data_module.train_dataloader())
