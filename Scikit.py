import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import xgboost as xgb
from Emoji_Link_Strip import clean_text
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def model_1():
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    #print(df[["text", "cleaned_text"]].head(10))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    vectorizer = TfidfVectorizer(lowercase= True, min_df = 2)
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])
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
    model = LogisticRegressionCV(max_iter= 5000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def model_2():
    # SVC, XGBoost, RandomForest
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf= True, strip_accents="ascii")),
    ("svc", LinearSVC(C=.6, penalty="l1", fit_intercept= True, intercept_scaling=10, class_weight={0: 1.0, 1: 1.15}))
])
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
    pipe.fit(train_df["cleaned_text"], y_train)
    preds = pipe.predict(test_df["cleaned_text"])
    print(classification_report(y_test, preds))

def model_3(): 
    sex_df = pd.read_csv('edos_labelled_data.csv')
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
    model = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose = False)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def model_4():

    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)
    df["cleaned_text"] = df["text"].apply(clean_text)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    vectorizer = TfidfVectorizer(lowercase= True, sublinear_tf= True)
    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    X_test = vectorizer.transform(test_df["cleaned_text"])
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
    model = RandomForestClassifier(n_estimators=350,random_state=42, min_samples_split=4, criterion="gini")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def main():
    #print("Model1:\n")
    #model_1()
    #print("Model2:\n")
    #model_2()
    #print("Model3:\n")
    #model_3()
    print("Model4:\n")
    model_4()
main()