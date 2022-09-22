#%%
#Import libraries

import nltk
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_hub as hub
import tensorflow_text as text

from matplotlib import pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

os.chdir(os.path.realpath(os.path.dirname(__file__)))
tf.get_logger().setLevel('ERROR')
AUTOTUNE = tf.data.AUTOTUNE
TFHUB_PREPROCESSOR = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
TFHUB_ENCODER = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
#%%

# FUNCTIONS / UTILS

def clean_text(text: str) -> str:
    '''Basic text cleaning for links, newlins and special characters'''
    www = r'www[A-Za-z0-9./]+'
    https = r"https?://[A-Za-z0-9./]+"
    newline = r'\n'
    special_chars = r'\B([!@#%&-]+)\s'
    for pattern in [www, https, newline, special_chars]:
        text = re.sub(pattern, '', text)
    return text


def binary_encode_label(label: int) -> tf.Tensor:
    if label == 0:
        return tf.constant([1, 0])
    else:
        return tf.constant([0, 1])

def build_tf_ds(dataframe: pd.DataFrame, batch_size: int = 32, is_training: bool = True) -> tf.data.Dataset:
    ds = (dataframe.review.to_numpy(), dataframe.label.to_numpy())
    ds = tf.data.Dataset.from_tensor_slices(ds)
    ds = ds.map(lambda review, label: (review, label))
    if is_training:
        ds = ds.shuffle(100)
    ds = ds.batch(batch_size).cache().prefetch(buffer_size=AUTOTUNE)
    return ds

def get_class_weights(labels: pd.Series) -> dict:
    class_weights = dict(1/(labels.value_counts()/len(labels)))
    return class_weights

def make_predictions(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    model_preds = model.predict(dataset)
    sig_preds= tf.sigmoid(model_preds)
    pred_labels = tf.map_fn(lambda x: 1 if x > 0.5 else 0, sig_preds)
    return pred_labels

def print_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> None:
    f1 = f1_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    roc_auc = roc_auc_score(true_labels, pred_labels)
    print(
        f"""
        F1 Score: {f1:.2f}
        Precision: {precision:.2f}
        Recall: {recall:.2f}
        ROC-AUC: {roc_auc:.2f}
        """
    )

def plot_confusion_matrix(
    true_labels: np.ndarray, 
    pred_labels: np.ndarray,
    labels: list = ["Positive Review", "Negative Review"]
):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_fig = ConfusionMatrixDisplay(cm, display_labels=labels)
    cm_fig.plot()
    return cm_fig


#%%

## EXPLORATORY DATA ANALYSIS

#After loading the data I check and notice that there is a missing review.
#I drop the data point because there is no information
#gained from a null review to label mapping and then clean the text.
reviews = pd.read_csv("review_data.csv")
reviews.count()
reviews.dropna(inplace=True)
reviews["review"] = reviews["review"].apply(clean_text)

#Notice there is a large class imbalance. We mostly get positive reviews with just the 87
#negative reviews. This will be important later. 
reviews["label"].hist()
reviews["label"].value_counts()

#%%

#Here I took a quick exploration of review length to see if there is anything interesting.
#I make a quick and dirty split into tokens and note that most reviews are shorter, although
#there are a couple lengthy positive reviews, but overall nothing too substantial to work with.
reviews["review_length"] = reviews["review"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

reviews["review_length"].hist(ax=axes[0][0])
reviews.plot(kind="scatter", x="label", y="review_length", ax=axes[0][1])
reviews[reviews["label"]==0]["review_length"].hist(ax=axes[1][0])
reviews[reviews["label"]==1]["review_length"].hist(ax=axes[1][1])

titles = [
    "All Review Lengths",
    "Review Length by Label",
    "Positive Review Lengths",
    "Negative Review Lengths"
]
for ax, title in zip(axes.ravel(), titles):
    ax.set_title(title)

reviews = reviews.drop("review_length", axis=1)

# %%

## DATA PREPARATION AND MODEL BUILDING

#First let's split the train and test to help judge whether I might be overfitting. 
#I stratify to keep the proportions of the classes the same in the train and test split

train, test = train_test_split(
    reviews,
    test_size = 0.2,
    stratify = reviews["label"]
)

#Load the data into tensorflow Datasets.
train_ds = build_tf_ds(train)
val_ds = build_tf_ds(test, is_training=False)

#%%
#We instantiate the model here. I use a max_seq_length of 900 somewhat heuristically. 
#As we saw when exploring the data, most reviews do not exceed 900 tokens or so and those are all
#positive reviews (class 0) and we care more about finding the negative reviews.
#This should help with training, but if I had more time I would test this a bit more.
#Otherwise I use a somewhat arbitrary learning rate with Adam and a Binary Cross entropy
#loss as is standard. I measure F1 Score, Recall, and Accuracy for comparison. 

class BertClassifier(tf.keras.Model):
    '''
    Basic Classifier model using BERT
    This class includes the preprocessing necessary to convert text into the tokens that 
    BERT expects. After that it only uses two linear layers with dropout in between to keep
    the model simple for now.
    '''
    def __init__(
        self, 
        num_classes: int, 
        units: int = 200,
        dropout_rate: float = 0.1,
        train_BERT: bool = False,
        preprocessor_handle: str = TFHUB_PREPROCESSOR,
        encoder_handle: str = TFHUB_ENCODER
    ):
        super(BertClassifier, self).__init__(name="prediction")
        self.preprocessor = hub.KerasLayer(preprocessor_handle)
        self.encoder = hub.KerasLayer(encoder_handle, trainable=train_BERT)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(units=units, activation="relu")
        self.output_layer = tf.keras.layers.Dense(num_classes)


    def call(self, text):
        processed_text = self.preprocessor(text)
        encoder_outputs = self.encoder(processed_text)
        pooled_output = encoder_outputs["pooled_output"]
        x = self.dropout(pooled_output)
        x = self.dense(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

model = BertClassifier(num_classes=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = [
    tfa.metrics.F1Score(num_classes=1), 
    tf.keras.metrics.Recall(), 
    tf.keras.metrics.Precision(),
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives()
]

model.compile(
    optimizer=optimizer,
    loss=loss,
    metrics=metrics
)
#%%

## MODEL TRAINING AND EVALUATION

#Here I fit the data with just two epochs (it takes a while to run on my laptop already).
#I also am only fine-tuning the final two dense layers, which might be improved if we allow 
#the gradients to flow through the pre-trained BERT model and adjust those weights a bit, but
#trying to train BERT was too much for my computer. I also adjust the training to weight the
# 1-class more by their prevalence in the whole dataset to help the training of the imbalanced
# dataset a bit.

history = model.fit(
    train_ds,
    validation_data=val_ds, 
    epochs=3, 
    class_weight=get_class_weights(reviews.label)
)

#Although, it is a short history, in general, the model is improving and has not really reached
#a plateau, which means that it could simply benefit from more training without increasing the 
#complexity to see improvements in performance. 
print(history.history)

#%%
#Let us make some predictions on the test set and take a look at a few metrics and the 
#confusion matrix

pred_labels = make_predictions(model, val_ds)
true_labels = test["label"].to_numpy()
print_metrics(true_labels, pred_labels)

#%%
cm_fig = plot_confusion_matrix(true_labels, pred_labels)

#%%

#For baseline comparison here is a pre-trained sentiment analyzer from NLTK:

class SentimentAnalyzer():
    def __init__(self):
        nltk.download('vader_lexicon')
        self.analyzer = SentimentIntensityAnalyzer()
    def predict(self, x):
        labels = []
        for text in x:
            score = self.analyzer.polarity_scores(text)["compound"]
            #Note giving inverted classes to match the customer review data where 1 is negative
            if score < 0.5:
                labels.append(1)
            else:
                labels.append(0)
        return labels

analyzer = SentimentAnalyzer()
reviews["nltk_labels"] = analyzer.predict(reviews["review"])

print_metrics(reviews["label"], reviews["nltk_labels"])
nltk_cm = plot_confusion_matrix(reviews["label"], reviews["nltk_labels"])

#%%

# #Some code for trying the holdout dataset:

# HOLDOUT_FILENAME = ""
# holdout = pd.read_csv(HOLDOUT_FILENAME)
# holdout["review"] = holdout["review"].apply(clean_text)
# holdout_ds = build_tf_ds(holdout, is_training=False)

# pred_labels = make_predictions(model, holdout_ds)
# true_labels = holdout["label"].to_numpy()
# print_metrics(true_labels, pred_labels)
# #%%
# cm_fig = plot_confusion_matrix(true_labels, pred_labels)