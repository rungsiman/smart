from transformers import AutoTokenizer


class CustomTokenizer(object):
    def __init__(self, experiment):
        self.experiment = experiment
    
    def encode(self, text, max_len):
        ...


class CustomAutoTokenizer(CustomTokenizer):
    def __init__(self, experiment, location=None):
        super().__init__(experiment)
        if experiment.lowercase is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(location or experiment.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(location or experiment.model, do_lower_case=experiment.lowercase)

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
