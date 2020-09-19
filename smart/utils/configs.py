import re


def override(bert_config, config):
    config_dict = {
        'hidden_dropout_prob': config.bert.hidden_dropout_prob,
        'g_hidden_dropout_prob': config.gan.generator.hidden_dropout_prob,
        'd_hidden_dropout_prob': config.gan.discriminator.hidden_dropout_prob,
        'g_noise_size': config.gan.generator.noise_size,
        'seq_classif_dropout': config.bert.seq_classif_dropout
    }

    bert_config.update(config_dict)


def select(kwargs, *classes, reverse=False):
    if reverse:
        return {key: value for key, value in kwargs.items() if all(not key.startswith(cls) for cls in classes)}
    else:
        filtered_kwargs = {}

        for cls in classes:
            for key, value in kwargs.items():
                if key.startswith(cls):
                    filtered_kwargs[re.sub(rf'^{cls}-', '', key)] = value

        return filtered_kwargs
