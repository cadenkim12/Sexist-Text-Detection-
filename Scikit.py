import pandas as pd
from sklearn.pipeline import Pipeline
import numpy as np
import xgboost as xgb
from Emoji_Link_Strip import clean_text
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def make_row(feature_model, y_test, preds):
    report = classification_report(y_test, preds, output_dict=True)

    row = {
        "Feature + Model": feature_model,
        "Sexist (P)": report["1"]["precision"],
        "Sexist (R)": report["1"]["recall"],
        "Sexist (F1)": report["1"]["f1-score"],
        "Non-Sexist (P)": report["0"]["precision"],
        "Non-Sexist (R)": report["0"]["recall"],
        "Non-Sexist (F1)": report["0"]["f1-score"],
        "Weighted (P)": report["weighted avg"]["precision"],
        "Weighted (R)": report["weighted avg"]["recall"],
        "Weighted (F1)": report["weighted avg"]["f1-score"]
    }

    return row

def model_1a():
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    #print(df[["text", "cleaned_text"]].head(10))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    vectorizer = CountVectorizer()
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
    return make_row("CountVectorizer + Logistic Regression", y_test, preds)

def model_1b():
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    #print(df[["text", "cleaned_text"]].head(10))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    vectorizer = CountVectorizer()
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
    model = LinearSVC(max_iter= 5000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return make_row("CountVectorizer + Linear SVC", y_test, preds)

def model_1c():
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    #print(df[["text", "cleaned_text"]].head(10))

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    vectorizer = CountVectorizer()
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

    # fixed: tree is a module, so this needs the classifier class
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return make_row("CountVectorizer + Decision Tree", y_test, preds)

def model_2a():
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
    return make_row("TF-IDF + LinearSVC", y_test, preds)

def model_2b():
    # SVC, XGBoost, RandomForest
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf= True, strip_accents="ascii")),
    ("xgb", xgb.XGBClassifier(eval_metric="logloss"))
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
    return make_row("TF-IDF + XGBoost", y_test, preds)

def model_2c():
    # SVC, XGBoost, RandomForest
    sex_df = pd.read_csv('edos_labelled_data.csv')
    df = pd.DataFrame(sex_df)

    df["cleaned_text"] = df["text"].apply(clean_text)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf= True, strip_accents="ascii")),
    ("rf", RandomForestClassifier())
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
    return make_row("TF-IDF + RandomForest", y_test, preds)


def main():
    results = []

    results.append(model_1a())
    results.append(model_1b())
    results.append(model_1c())
    results.append(model_2a())
    results.append(model_2b())
    results.append(model_2c())

    labels = ["Feature + Model", "Sexist (P)", "Sexist (R)", "Sexist (F1)", "Non-Sexist (P)", "Non-Sexist (R)", 
              "Non-Sexist (F1)", "Weighted (P)", "Weighted (R)", "Weighted (F1)" ]

    df = pd.DataFrame(results, columns=labels)
    df = df.round(2)

    print(df.to_string(index=False))

main()
