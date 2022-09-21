#%%
#Import libraries

import bert
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa

from functools import partial
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, 
    InputExample,
    InputFeatures,
    TFBertForSequenceClassification, 
)

os.chdir(os.path.realpath(os.path.dirname(__file__)))
tf.get_logger().setLevel('ERROR')
AUTOTUNE = tf.data.AUTOTUNE
TFHUB_PREPROCESSOR = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
TFHUB_ENCODER = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
#%%

## Exploratory Data Analysis

reviews = pd.read_csv("review_data.csv")
reviews.count()
#There is a missing review and so I drop the data point because there is no information
#gained from a null review to label mapping.
reviews.dropna(inplace=True)

#Notice there is a large class imbalance. We mostly get positive reviews with just the 87
#negative reviews. 
reviews["label"].hist()
reviews["label"].value_counts()

#%%

#Take a quick look at review length
reviews["review_length"] = reviews["review"].apply(lambda x: len(str(x).split()))

#Most reviews are shorter as expected, there are a couple lengthy positive reviews,
#but nothing that substantial here to work with.

length_plot, length_axes = plt.subplots(2, 2, figsize=(12, 12))

reviews["review_length"].hist(ax=length_axes[0][0])
length_axes[0][0].set_title("All Review Lengths")

reviews.plot(kind="scatter", x="label", y="review_length", ax=length_axes[0][1])
length_axes[0][1].set_title("Review Length by Label")

reviews[reviews["label"]==0]["review_length"].hist(ax=length_axes[1][0])
length_axes[1][0].set_title("Positive Review Lengths")

reviews[reviews["label"]==1]["review_length"].hist(ax=length_axes[1][1])
length_axes[1][1].set_title("Negative Review Lengths")
#%%
reviews = reviews.drop("review_length", axis=1)
# %%
reviews["review"] = reviews["review"].apply(lambda x: re.sub(r'\n', ' ', x))
# %%

#Stratify to help adjust for class imbalance

train, test = train_test_split(
    reviews,
    test_size = 0.2,
    stratify = reviews["label"]
)
#%%
class Classifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(Classifier, self).__init__(name="prediction")
        self.preprocessor = hub.KerasLayer(TFHUB_PREPROCESSOR)
        self.encoder = hub.KerasLayer(TFHUB_ENCODER, trainable=False)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.dense = tf.keras.layers.Dense(num_classes)

    def call(self, text):
        preprocessed_text = self.preprocessor(text)
        encoder_outputs = self.encoder(preprocessed_text)
        pooled_output = encoder_outputs["pooled_output"]
        x = self.dropout(pooled_output)
        x = self.dense(x)
        return x

def binary_encode_label(label: int) -> tf.Tensor:
    if label == 0:
        return tf.constant([1, 0])
    else:
        return tf.constant([0, 1])

def build_tf_ds(dataframe: pd.DataFrame, batch_size: int = 32, is_training: bool = True) -> tf.data.Dataset:
    ds = (dataframe.review.to_numpy(), dataframe.label.to_numpy())
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(lambda review, label: (review, binary_encode_label(label)))
    if is_training:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    return ds

def get_class_weights(labels: pd.Series) -> dict:
    class_weights = dict(1/(labels.value_counts()/len(labels)))
    return class_weights

train_ds = build_tf_ds(train)
val_ds = build_tf_ds(test)

#%%
model = Classifier(2)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [tfa.metrics.F1Score(num_classes=2), tf.keras.metrics.Recall()]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
#%%
history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=2, 
    class_weight=get_class_weights(reviews.label)
)
#%%