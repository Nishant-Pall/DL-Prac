# Specify how preprocessing should be done, using Fields
# Use dataset to load the data, TabularDataset (JSON/CSV/TSV) files
# Construct an iterator to do batching and padding using BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator
import spacy

spacy_en = spacy.load('en_core_web_sm')


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {
    'quote': ('q', quote), 'score': ('s', score)
}

train_data, test_data = TabularDataset.splits(
    path='.',
    train='train.json',
    #     train='train.csv',
    #     train='train.tsv',
    test='test.json',
    #     test='test.csv',
    #     test='test.tsv',
    # validate = 'validation.json',
    format='json',
    #     format='csv',
    #     format='tsv',
    fields=fields
)

quote.build_vocab(train_data, max_size=10000,
                  min_freq=1,
                  #  vectors='gLove.6B.100d'
                  )

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2
)
