from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    pretrained_model_name: str = "monologg/koelectra-base-v3-discriminator"

    context_max_length: int = 512
    context_padding: str = "longest"

    question_max_length: int = 512
    question_padding: str = "longest"


settings = Settings()
