from datasets import load_dataset
from datasets.utils.typing import PathLike
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import ElectraTokenizer

from re2g.configs import settings

DATALOADER_NUM_WORKERS = settings.dataloader_num_workers

QUERY_MAX_LENGTH = settings.query_max_length

QUERY_PADDING = settings.query_padding

CONTEXT_MAX_LENGTH = settings.context_max_length

CONTEXT_PADDING = settings.context_padding

RERANK_MAX_LENGTH = settings.rerank_max_length

RERANK_PADDING = settings.rerank_padding


class SquadDataset(Dataset):
    def __init__(self, data_type: str = "train"):
        self.dataset = load_dataset("squad_kor_v1")[data_type]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        # item shape will be
        # {'id': '6566495-0-0', 'title': '파우스트_서곡', 'context': '1839년 바그너는 괴테의 파우스트을 처음 읽고 그 내용에 마음이 끌려 이를 소재로 해서 하나의 교향곡을 쓰려는 뜻을 갖는다. 이 시기 바그너는 1838년에 빛 독촉으로 산전수전을 다 걲은 상황이라 좌절과 실망에 가득했으며 메피스토펠레스를 만나는 파우스트의 심경에 공감했다고 한다. 또한 파리에서 아브네크의 지휘로 파리 음악원 관현악단이 연주하는 베토벤의 교향곡 9번을 듣고 깊은 감명을 받았는데, 이것이 이듬해 1월에 파우스트의 서곡으로 쓰여진 이 작품에 조금이라도 영향을 끼쳤으리라는 것은 의심할 여지가 없다. 여기의 라단조 조성의 경우에도 그의 전기에 적혀 있는 것처럼 단순한 정신적 피로나 실의가 반영된 것이 아니라 베토벤의 합창교향곡 조성의 영향을 받은 것을 볼 수 있다. 그렇게 교향곡 작곡을 1839년부터 40년에 걸쳐 파리에서 착수했으나 1악장을 쓴 뒤에 중단했다. 또한 작품의 완성과 동시에 그는 이 서곡(1악장)을 파리 음악원의 연주회에서 연주할 파트보까지 준비하였으나, 실제로는 이루어지지는 않았다. 결국 초연은 4년 반이 지난 후에 드레스덴에서 연주되었고 재연도 이루어졌지만, 이후에 그대로 방치되고 말았다. 그 사이에 그는 리엔치와 방황하는 네덜란드인을 완성하고 탄호이저에도 착수하는 등 분주한 시간을 보냈는데, 그런 바쁜 생활이 이 곡을 잊게 한 것이 아닌가 하는 의견도 있다.', 'question': '바그너는 괴테의 파우스트를 읽고 무엇을 쓰고자 했는가?', 'answers': {'text': ['교향곡'], 'answer_start': [54]}}
        return self.dataset[idx]


class DprSquadDataModule(LightningDataModule):
    def __init__(
        self, pretrained_model_name_or_path: str | PathLike, batch_size: int = 128
    ):
        super().__init__()
        self.batch_size = batch_size
        self.squad_train = SquadDataset(data_type="train")
        self.squad_val = SquadDataset(data_type="validation")
        self.squad_test = SquadDataset(data_type="validation")
        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)

        # self.squad_train.dataset = self.squad_train.dataset[:100]
        self.squad_val.dataset.shuffle()
        self.squad_test.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(
            self.squad_train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.squad_val,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.squad_test,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def _collate_fn(self, batch: list[dict]):
        context_batch_encoding = self.tokenizer.batch_encode_plus(
            [(item["context"]) for item in batch],
            max_length=CONTEXT_MAX_LENGTH,
            padding=CONTEXT_PADDING,
            truncation=True,
            return_tensors="pt",
        )
        query_batch_encoding = self.tokenizer.batch_encode_plus(
            [(item["question"]) for item in batch],
            max_length=QUERY_MAX_LENGTH,
            padding=QUERY_PADDING,
            truncation=True,
            return_tensors="pt",
        )
        contexts = [item["context"] for item in batch]
        queries = [item["question"] for item in batch]
        return {
            "query_input_ids": query_batch_encoding["input_ids"],
            "query_attention_mask": query_batch_encoding["attention_mask"],
            "query_token_type_ids": query_batch_encoding["token_type_ids"],
            "queries": queries,
            "context_input_ids": context_batch_encoding["input_ids"],
            "context_attention_mask": context_batch_encoding["attention_mask"],
            "context_token_type_ids": context_batch_encoding["token_type_ids"],
            "contexts": contexts,
        }


class RerankSquadDataModule(LightningDataModule):
    def __init__(
        self, pretrained_model_name_or_path: str | PathLike, batch_size: int = 128
    ):
        super().__init__()
        self.batch_size = batch_size
        self.squad_train = SquadDataset(data_type="train")
        self.squad_val = SquadDataset(data_type="validation")
        self.squad_test = SquadDataset(data_type="validation")
        self.tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)

        # self.squad_train.dataset = self.squad_train.dataset[:100]
        self.squad_val.dataset.shuffle()
        self.squad_test.dataset.shuffle()

    def train_dataloader(self):
        return DataLoader(
            self.squad_train,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            self.squad_val,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            self.squad_test,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
        )

    def _collate_fn(self, batch: list[dict]):
        batch_encoding = self.tokenizer.batch_encode_plus(
            [(item["question"], item["context"]) for item in batch],
            max_length=CONTEXT_MAX_LENGTH,
            padding=CONTEXT_PADDING,
            truncation=True,
            return_tensors="pt",
        )
        queries = [item["question"] for item in batch]
        contexts = [item["context"] for item in batch]
        return {
            "input_ids": batch_encoding["input_ids"],
            "attention_mask": batch_encoding["attention_mask"],
            "token_type_ids": batch_encoding["token_type_ids"],
            "queries": queries,
            "contexts": contexts,
        }
