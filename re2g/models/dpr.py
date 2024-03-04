import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities import grad_norm
from torch import nn
from transformers import ElectraModel


class QueryEncoder(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(QueryEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class ContextEncoder(nn.Module):
    def __init__(
        self, pretrained_model_name_or_path: str, num_trainable_layers: int = 2
    ):
        super(ContextEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

        for param in self.electra.parameters():
            param.requires_grad = False

        for count, layer in enumerate(
            self.electra.encoder.layer[-num_trainable_layers:]
        ):
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class DPR(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        num_query_trainable_layers: int = 2,
        num_context_trainable_layers: int = 2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        super(DPR, self).__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.query_encoder = QueryEncoder(
            pretrained_model_name_or_path,
            num_trainable_layers=num_query_trainable_layers,
        )
        self.context_encoder = ContextEncoder(
            pretrained_model_name_or_path,
            num_trainable_layers=num_context_trainable_layers,
        )
        self.criteria = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ):
        query_embeddings = self.query_encoder(query_input_ids, query_attention_mask)
        context_embeddings = self.context_encoder(
            context_input_ids, context_attention_mask
        )
        return query_embeddings, context_embeddings

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        batch_size = query_input_ids.shape[0]

        query_embeddings, context_embeddings = self.forward(
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
        )
        loss = self.loss(query_embeddings, context_embeddings)
        self.log(
            name="train_loss",
            value=loss,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        query_input_ids = batch["query_input_ids"]
        query_attention_mask = batch["query_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        batch_size = query_input_ids.shape[0]

        query_embeddings, context_embeddings = self.forward(
            query_input_ids,
            query_attention_mask,
            context_input_ids,
            context_attention_mask,
        )
        loss = self.loss(query_embeddings, context_embeddings)
        mrr = self.calculate_mrr(query_embeddings, context_embeddings)
        self.log(
            name="val_loss",
            value=loss,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            name="mrr",
            value=mrr,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return {"val_loss": loss, "mrr": mrr}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        """Calculate Mean Reciprocal Rank (MRR)"""
        pass

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self.layer, norm_type=2)
    #     self.log_dict(norms)
    # trainer = Trainer(detect_anomaly=True)

    def loss(
        self,
        query_embeddings: torch.Tensor,
        context_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        query_embeddings_t = query_embeddings.transpose(0, 1)
        similarity_scores = torch.matmul(context_embeddings, query_embeddings_t)
        labels = torch.arange(
            similarity_scores.size(0),
            device=query_embeddings.device,
            dtype=torch.long,
        )
        loss = self.criteria(similarity_scores, labels)
        return loss

    @staticmethod
    def calculate_mrr(
        context_embeddings: torch.Tensor, query_embeddings: torch.Tensor
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR)"""

        similarity_scores = torch.matmul(context_embeddings, query_embeddings.t())
        similarity_scores = similarity_scores.cpu().detach().numpy()

        acc = 0.0
        for i in range(similarity_scores.shape[0]):
            scores = similarity_scores[i]
            scores = np.argsort(-scores)
            for j, rank in enumerate(scores):
                if rank == i:
                    acc += 1 / (j + 1)
                    break
        m = acc / similarity_scores.shape[0]
        return m
