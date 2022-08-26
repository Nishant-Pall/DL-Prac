import pandas as pd
import spacy
from torchtext.data import Field, BucketIerator, TabularDataset
from sklearn.model_selection import train_test_split

english_txt = open("train_eng.txt", encoding='utf-8').read().split('\n')
german_txt = open("german_eng.txt", encoding='utf-8').read().split('\n')

raw_data = {
    "English": [line for line in english_txt[1:10000]],
    "German": [line for line in german_txt[1:10000]]
}

df = pd.DataFrame(raw_data, columns=["English", "German"])

train_data, test_data = train_test_split(df, test_size=0.2)

train_data.to_json("train.json", orient="records", lines=True)
test_data.to_json("test.json", orient="records", lines=True)

train_data.to_csv("train.csv", index=False)
test_data.to_csv("test.csv", index=False)
