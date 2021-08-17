#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter


# In[2]:


spacy_ger = spacy.load('de_core_news_sm')
spacy_eng = spacy.load('en_core_web_sm')


# In[3]:


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# In[4]:


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# In[5]:


# FIELD FOR DEFINING PREPROCESSING

german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='<sos>', eos_token='<eos>')


english = Field(tokenize=tokenizer_eng, lower=True,
                init_token='<sos>', eos_token='<eos>')


# In[6]:


train_data, validation_data, test_data = Multi30k(
    split=('train', 'valid', 'test'), language_pair=('de', 'en'))


# In[7]:


german.build_vocab(train_data, max_size=10000, min_freq=1)
english.build_vocab(train_data, max_size=10000, min_freq=1)


# In[8]:


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape :(seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))
        # shape: (seq_length, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


# In[9]:


class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x : (batch_size) but we want (1, batch_size)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # shape; (1, batch_size, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs: (1, batch_size, hidden_size)

        predictions = self.fc(outputs)
        # predictions: (1, batch_size, length_of_vocab)

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


# In[10]:


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        # source: (target_len, batch_size)
        target_len = target.shape[0]
        target_vocab_size = len(german.vocab)

        outputs = torch.zeros(target_len, batch_size,
                              target_vocab_size).to(device)
        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            # output : (batch_size, german_vocab_size)
            best_guess = output.argmax(1)

            x = target[t] if random.random(
            ) < teacher_force_ratio else best_guess

        return outputs


# In[11]:


# TRAINING

num_epochs = 20
learning_rate = 0.001
batch_size = 64

load_model = False
input_size_encoder = len(english.vocab)
input_size_decoder = len(german.vocab)
output_size = len(german.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 128
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits((train_data, validation_data, test_data),
                                                                           batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src))


# In[15]:


encoder_net = Encoder(input_size_encoder, encoder_embedding_size,
                      hidden_size, num_layers, enc_dropout)


# In[16]:


decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size,  num_layers, dec_dropout)


# In[17]:


model = Seq2Seq(encoder_net, decoder_net)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# In[18]:


pad_idx = german.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)


# In[22]:


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(
    ), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


# In[ ]:
