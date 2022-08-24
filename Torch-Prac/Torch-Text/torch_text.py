# Specify how preprocessing should be done, using Fields
# Use dataset to load the data, TabularDataset (JSON/CSV/TSV) files
# Construct an iterator to do batching and padding using BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator


def tokenize(x): return x.split()


quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {
    'quote': ('q', quote), 'score': ('s', score)
}

train_data, test_data = TabularDataset.splits(
    path='.',
    train='train.json',
    test='test.json',
    # validate = 'validation.json',
    format='json',
    fields=fields
)

# train_data, test_data = TabularDataset.splits(
#     path='data',
#     train='train.csv',
#     test='test.csv',
#     format='csv',
#     fields=fields
# )

# train_data, test_data = TabularDataset.splits(
#     path='data',
#     train='train.tsv',
#     test='test.tsv',
#     format='tsv',
#     fields=fields
# )

print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())
