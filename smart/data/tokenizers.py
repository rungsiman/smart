import abc
from transformers import AutoTokenizer


class CustomTokenizer(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        self.config = config

    @abc.abstractmethod
    def encode(self, text, max_len):
        ...

    @abc.abstractmethod
    def find_max_len(self, texts):
        ...

    @abc.abstractmethod
    def save(self, target_location):
        ...


class CustomAutoTokenizer(CustomTokenizer):
    def __init__(self, config, location=None):
        super().__init__(config)
        if config.lowercase is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(location or config.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(location or config.model, do_lower_case=config.lowercase)

    def encode(self, text, max_len):
        return self.tokenizer.encode_plus(text,
                                          padding='max_length',
                                          max_length=max_len,
                                          truncation=True,
                                          add_special_tokens=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=True,
                                          return_tensors='pt')

    def find_max_len(self, texts):
        max_len = 0

        for text in texts:
            if text is not None:
                input_ids = self.tokenizer.encode(text, add_special_tokens=True)
                max_len = max(max_len, len(input_ids))
        
        return max_len

    def save(self, target_location):
        self.tokenizer.save_vocabulary(target_location)
        return self
