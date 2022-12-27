from transformers import PretrainedConfig


class CustomDPRConfig(PretrainedConfig):
    model_type = 'dpr'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)