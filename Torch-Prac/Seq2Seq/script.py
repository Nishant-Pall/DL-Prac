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
