import lightning as L
import torch
from torch import nn
from transformers import ElectraModel


class ReRanker(nn.Module):

    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(ReRanker, self).__init__()

        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)
        self.linear_1 = nn.Linear(768, 16)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        x = sequence_output[:, 0, :]
        x = self.linear_1(x)
        x = self.tanh(x)
        x = self.linear_2(x)
        x = self.sigmoid(x)
        return x


class Rerank(L.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_trainable_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(Rerank, self).__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.num_trainable_layers = num_trainable_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.reranker = ReRanker(pretrained_model_name_or_path, num_trainable_layers)
        self.save_hyperparameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ):
        scores = self.reranker(input_ids, attention_mask)
        return scores

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        shape = input_ids.shape

        input_ids = input_ids.view(-1, shape[-1])
        attention_mask = attention_mask.view(-1, shape[-1])
        token_type_ids = token_type_ids.view(-1, shape[-1])

        scores = self.forward(input_ids, attention_mask, token_type_ids)
        scores = scores.view(labels.shape)

        loss = nn.BCELoss()(scores, labels.float())
        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=shape[0],
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        labels = batch["labels"]
        shape = input_ids.shape

        input_ids = input_ids.view(-1, shape[-1])
        attention_mask = attention_mask.view(-1, shape[-1])
        token_type_ids = token_type_ids.view(-1, shape[-1])

        scores = self.forward(input_ids, attention_mask, token_type_ids)
        scores = scores.view(labels.shape)

        loss = nn.BCELoss()(scores, labels.float())
        self.log(
            name="val_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=shape[0],
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
