import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score

def model_1():
    sex_df = pd.read_csv('final_project/edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train = []
    for bool in train_df["label"]:
        if bool == "sexist":
            y_train.append(1) 
        else:
            y_train.append(0)
    y_test = []
    for bool in test_df["label"]:
        if bool == "sexist":
            y_test.append(1)
        else:
            y_test.append(0)
    model = LogisticRegressionCV()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    weighted_f1 = f1_score(y_test, preds, average="weighted")
    print(weighted_f1)
    print(preds)
    print(model.score(X_test, y_test))

def model_2():
    sex_df = pd.read_csv('final_project/edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    train_texts = train_df["text"]
    y_train = []
    for bool in train_df["label"]:
        if bool == "sexist":
            y_train.append(1) 
        else:
            y_train.append(0)
    y_test = []
    for bool in test_df["label"]:
        if bool == "sexist":
            y_test.append(1)
        else:
            y_test.append(0)
    vectorize = tf.keras.layers.TextVectorization(
        max_tokens = 5000,
        output_mode ="int",
        output_sequence_length = 10)
    
    vectorize.adapt(train_texts)

    # Build Neural Networkd
    #1. Build Model
    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
        vectorize,
])
    #2. Compile the model
    model.compile(loss =tf.keras.losses.mae,
                        optimizer = tf.keras.optimizers.SGD(),
                        metrics=["mae"])
    # USED AI
    X_train = train_df["text"].astype(str).to_numpy()
    X_test = test_df["text"].astype(str).to_numpy()

    print(X_train)
    print(X_test)
    #3. Fit the model
    model.fit(X_train, y_train,epochs=100)

def main():
    model_1()
    model_2()
main()