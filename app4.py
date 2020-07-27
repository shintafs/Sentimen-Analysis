import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

dataframe = pd.read_csv("dataset2k.txt")

x=dataframe.iloc[:,0]
y=dataframe.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')

kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline_svm = make_pipeline(vect,
                            SVC(probability=True, kernel="linear", class_weight="balanced"))

grid = GridSearchCpyV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]},
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,
                    n_jobs=-1)

grid.fit(x_train, y_train)
print grid.score(x_test, y_test)