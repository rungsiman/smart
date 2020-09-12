def override(bert_config, config):
    config_dict = {
        'hidden_dropout_prob': config.bert.hidden_dropout_prob,
        'g_hidden_dropout_prob': config.gan.generator.hidden_dropout_prob,
        'd_hidden_dropout_prob': config.gan.discriminator.hidden_dropout_prob,
        'g_noise_size': config.gan.generator.noise_size,
        'seq_classif_dropout': config.bert.seq_classif_dropout
    }

    bert_config.update(config_dict)
