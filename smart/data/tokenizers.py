from transformers import AutoTokenizer


class CustomTokenizer(object):
    def __init__(self, experiment):
        self.experiment = experiment
    
    def encode(self, text, max_len):
        ...


class CustomAutoTokenizer(CustomTokenizer):
    def __init__(self, experiment):
        super().__init__(experiment)
        self.tokenizer = AutoTokenizer.from_pretrained(experiment.model)

    def encode(self, text, max_len):
        return self.tokenizer.encode_plus(text,
                                          max_length=max_len,
                                          truncation=True,
                                          add_special_tokens=True,
                                          pad_to_max_length=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=True,
                                          return_tensors='pt')

    def find_max_len(self, texts):
        max_len = 0

        for text in texts:
            input_ids = self.tokenizer.encode(text, add_special_tokens=True)
            max_len = max(max_len, len(input_ids))
        
        return max_len
