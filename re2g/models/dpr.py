import lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from transformers import ElectraModel


class QueryEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super(QueryEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class ContextEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str):
        super(ContextEncoder, self).__init__()
        self.electra = ElectraModel.from_pretrained(pretrained_model_name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        return cls_token_output


class DPR(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path: str):
        super(DPR, self).__init__()
        self.query_encoder = QueryEncoder(pretrained_model_name_or_path)
        self.context_encoder = ContextEncoder(pretrained_model_name_or_path)

    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
    ):
        question_embeddings = self.query_encoder(
            question_input_ids, question_attention_mask
        )
        context_embeddings = self.context_encoder(
            context_input_ids, context_attention_mask
        )
        return question_embeddings, context_embeddings

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        question_input_ids = batch["question_input_ids"]
        question_attention_mask = batch["question_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        query_embeddings, context_embeddings = self(
            question_input_ids,
            question_attention_mask,
            context_input_ids,
            context_attention_mask,
        )
        loss = self.loss(query_embeddings, context_embeddings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        question_input_ids = batch["question_input_ids"]
        question_attention_mask = batch["question_attention_mask"]
        context_input_ids = batch["context_input_ids"]
        context_attention_mask = batch["context_attention_mask"]
        query_embeddings, context_embeddings = self(
            question_input_ids,
            question_attention_mask,
            context_input_ids,
            context_attention_mask,
        )
        loss = self.loss(query_embeddings, context_embeddings)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss}

    def loss(self, query_embeddings, context_embeddings):
        query_embeddings_t = query_embeddings.transpose(0, 1)
        similarity_scores = torch.matmul(context_embeddings, query_embeddings_t)
        labels = torch.arange(similarity_scores.size(0))
        labels = labels.long().to(query_embeddings.device)
        loss = F.cross_entropy(similarity_scores, labels)
        return loss
