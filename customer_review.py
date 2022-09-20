#%%
import os
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

os.chdir(os.path.realpath(os.path.dirname(__file__)))

#%%

reviews = pd.read_csv("review_data.csv")

#%%
reviews.count()
reviews.dropna(inplace=True)

reviews["label"].hist()
reviews["label"].value_counts()

#%%

reviews["review_length"] = reviews["review"].apply(lambda x: len(str(x)))

reviews["review_length"].hist()
reviews[reviews["label"]==0]["review_length"].hist()
reviews[reviews["label"]==1]["review_length"].hist()
plt.scatter(reviews["label"], reviews["review_length"])
# %%

x_train, x_test, y_train, y_test = train_test_split(reviews["review"], reviews["label"], test_size=0.2)

