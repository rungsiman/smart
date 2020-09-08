import abc
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss


class GanGenerator(nn.Module):
    def __init__(self, config, noise_size=100):
        super().__init__()
        self.dense = nn.Linear(noise_size, config.hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.noise_size = noise_size

    def forward(self, noise):
        hidden_states = self.dense(noise)
        hidden_states = self.leaky_relu(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class GanDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 1)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.leaky_relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        return logits, hidden_states


class GanForClassificationBase(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self, config):
        super().__init__()

        self.generator = GanGenerator(config)
        self.discriminator = GanDiscriminator(config)
        self.noise_size = config.g_noise_size
        self.num_labels = config.num_labels

    def forward(self, pooled_output, labels, epsilon=1e-8):
        d_real_logits, d_real_features = self.discriminator(pooled_output)

        if labels is not None:
            # The first labels.shape[0] sentences in a batch are for training (have labels)
            sup_size = labels.shape[0]

            d_real_probs = self._compute_probs(d_real_logits)

            # Remove the fake class since it does not contribute to the loss of supervised training
            d_real_sup_logits = d_real_logits[:sup_size, :-1]

            # Generate noise for every token
            noise = torch.rand((pooled_output.shape[0], self.noise_size), device=pooled_output.device)
            g_noise = self.generator(noise)
            d_fake_logits, d_fake_features = self.discriminator(g_noise)
            d_fake_probs = self._compute_probs(d_fake_logits)

            lg_feature = torch.mean(torch.square(torch.mean(d_real_features, dim=0) - torch.mean(d_fake_features, dim=0)))
            lg_unsup = -torch.log(1 - d_fake_probs[:, -1] + epsilon).mean()
            lg = lg_feature + lg_unsup

            ld_sup = self._compute_ld_sup(d_real_sup_logits, labels)
            ld_unsup_1 = -torch.log(1 - d_real_probs[:, -1] + epsilon).mean()
            ld_unsup_2 = -torch.log(d_fake_probs[:, -1] + epsilon).mean()
            ld = ld_sup + ld_unsup_1 + ld_unsup_2

            return ld, lg, d_real_logits[:, :-1]

        else:
            return None, None, d_real_logits[:, :-1]

    @abc.abstractmethod
    def _compute_probs(self, logits):
        ...

    @abc.abstractmethod
    def _compute_ld_sup(self, d_real_sup_logits, labels):
        ...


class GanForSequenceClassification(GanForClassificationBase):
    def _compute_probs(self, logits):
        return F.softmax(logits, dim=-1)

    def _compute_ld_sup(self, d_real_sup_logits, labels):
        loss_fct = CrossEntropyLoss()
        return loss_fct(d_real_sup_logits.view(-1, self.num_labels), labels.view(-1))


class GanForMultipleLabelClassification(GanForClassificationBase):
    def _compute_probs(self, logits):
        return torch.sigmoid(logits)

    def _compute_ld_sup(self, d_real_sup_logits, labels):
        loss_fct = BCEWithLogitsLoss()
        return loss_fct(d_real_sup_logits.view(-1, self.num_labels), labels.float().view(-1, self.num_labels))


class GanForPairedLabelClassification(GanForMultipleLabelClassification):
    def _compute_probs(self, logits):
        return torch.sigmoid(logits)

    def _compute_ld_sup(self, d_real_sup_logits, labels):
        loss_fct = MSELoss()
        return loss_fct(d_real_sup_logits.view(-1), labels.float().view(-1))
