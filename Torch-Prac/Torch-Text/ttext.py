import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

spacy_en = spacy.load('en_core_web_sm')
spacy_ger = spacy.load('de_core_web_sm')


def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


english = Field(sequential=True, use_vocab=True,
                tokenize=tokenize_eng, lower=True)
german = Field(sequential=True, use_vocab=True,
               tokenize=tokenize_ger, lower=True)

train_data, validation_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english))
