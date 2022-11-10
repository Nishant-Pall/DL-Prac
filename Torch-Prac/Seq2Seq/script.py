import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

spacy_eng = spacy.load('en_core_web_sm')
spacy_ger = spacy.load('de_core_news_sm')


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')
german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(
    fields=(english, german), exts=('.en', '.de'))

english.build_vocab(train_data, max_size=10000, min_freq=2)
german.build_vocab(train_data, max_size=10000, min_freq=2)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p) -> None:
        super(Encoder).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape : (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))
        # embedding shape : (seq_length, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p) -> None:
        super(Decoder).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)

        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):

        # to make x's shape from (N) to (1,N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))

        predictions = self.fc(outputs)

        # to make prediction's shape from (1, N, length_of_vocab) to (N, length_of_vocab)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Seq2Seq).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(german.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)

        hidden, cell = self.encoder(source)

        # grab <sos> token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            x = target[t] if random.random(
            ) < teacher_forcing_ratio else output.argmax(1)

        return outputs


# TRAINING

# Training hyperparams
num_epochs = 25
learning_rate = 0.001
batch_size = 64


# Model hyperparams
load_model = False
input_size_encoder = len(english.vocab)
input_size_decoder = len(german.vocab)
output_size = len(german.vocab)
encoder_embedding_size = 200
decoder_embedding_size = 200
hidden_size = 1024
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    datasets=(train_data, validation_data, test_data),
    batch_size=batch_size,
    # sort data with length
    sort_within_batch=True,
    sort_key=lambda x: len(x.src)
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, encoder_dropout)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, num_layers, decoder_dropout)
