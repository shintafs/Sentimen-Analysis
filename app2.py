import pandas as pd
from sklearn.pipeline import make_pipeline, Pipeline
from preprocessing import Preprocessing
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score
import datetime
import time


svm_dataset = pd.read_csv("dataset2k.txt", sep="\t",
                          header=None, encoding='cp1252', error_bad_lines=False)
svm_dataset.columns = ['text', 'label']
positives = svm_dataset['label'][svm_dataset.label == 1]
negatives = svm_dataset['label'][svm_dataset.label == 0]
COLNAMES = ["id", "text"]
nltk.download('stopwords')


def word_count(text):
    return len(str(text).split())


svm_dataset["word_count"] = svm_dataset["text"].apply(word_count)
print("Dataset loaded successfully!")

all_words = []
for line in list(svm_dataset['text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

dataset = svm_dataset

####
Start = time.time()
dataset.to_pickle("dataset.p")
dataset_pickle = pd.read_pickle("dataset.p")
dataset_pickle['text'] = dataset_pickle['text'].apply(
    Preprocessing().processTweet)
dataset_pickle_pickle = dataset_pickle.drop_duplicates('text')
dataset_pickle.shape

eng_stop_words = stopwords.words('indonesian')
dataset_pickle = dataset_pickle.copy()
dataset_pickle['tokens'] = dataset_pickle['text'].apply(
    Preprocessing().text_process)

bow_transformer = CountVectorizer(
    analyzer=Preprocessing().text_process).fit(dataset_pickle['text'])
messages_bow = bow_transformer.transform(dataset_pickle['text'])
print("Dataset dibersihkan!")
print("\nMulai train / test dengan perbandingan training 80% dan testing 20%")
# test nya hanya 20%, training nya 80%
X_train, X_test, y_train, y_test = train_test_split(
    svm_dataset['text'], svm_dataset['label'], test_size=0.2)

en_stopwords = set(stopwords.words("indonesian"))
vect = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b\w{1,}\b')
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

pipeline_svm = make_pipeline(vect,
                             TfidfTransformer(),
                             SVC(probability=True, kernel="rbf", class_weight="balanced"))

grid = GridSearchCV(pipeline_svm,
                    param_grid={'svc__C': [0.01, 0.1, 1]},
                    cv=kfolds,
                    scoring="roc_auc",
                    verbose=1,
                    n_jobs=-1)

grid.fit(X_train, y_train)


means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']


joblib.dump(grid, "model2.pkl")
# buat test model
model_SVM = joblib.load("model2.pkl")

y_preds = model_SVM.predict(X_test)

print('akurasi dari train/test split: ',
      str(accuracy_score(y_test, y_preds) * 100) + "%")
print('confusion matrix: \n', confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))

# testing
model_SVM = joblib.load("model2.pkl")


def label_to_str(x):
    if x == 0:
        return 'Negative'
    else:
        return 'Positive'


x = 0
text_ = [0] * len(svm_dataset)
label_ = [0] * len(svm_dataset)

for review in svm_dataset['text']:
    predict = model_SVM.predict([review])
    text_[x] = review
    label_[x] = predict[0]
    x += 1

print("write ke csv")
hehe = {"text": text_, "label": label_}
hehe2 = pd.DataFrame(data=hehe)
hehe2.to_csv('test_ulang_dataset_svm.csv', header=True, index=False,
             encoding='cp1252')
hasil_test_ulang = pd.read_csv(
    "test_ulang_dataset_svm.csv", encoding='cp1252', header='infer', error_bad_lines=False)
hasil_test_ulang.columns = ['text', 'label']

# recheck_pos = hasil_test_ulang['label'][hasil_test_ulang.label == "Positive"]
# recheck_neg = hasil_test_ulang['label'][hasil_test_ulang.label == "Negative"]
# print("Hasil test ulang punya positive prediksi sebanyak : "+ str(len(recheck_pos)) +" dan negatif sebanyak " + str(len(recheck_neg)))

i = 0
true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0
for predicted_label in hasil_test_ulang['label']:

    if svm_dataset['label'][i] == 1:
        if predicted_label == 1:
            true_positive += 1
        else:
            false_negative += 1

    if svm_dataset['label'][i] == 0:
        if predicted_label == 0:
            true_negative += 1
        else:
            false_positive += 1
    i += 1


Finish = time.time()
runningTime = Finish - Start
print("====== Hasil Sentimen Analisis SVM ======")
now = datetime.datetime.now()
print("Run Pada : " + now.strftime("%Y-%m-%d %H:%M:%S"))
print("True positive : " + str(true_positive))
print("True negative : " + str(true_negative))
print("False positive : " + str(false_positive))
print("False negative : " + str(false_negative))
print("Running Time --- %s seconds ---" % (runningTime))

# akurasi TP + TN / TP + FN + FP + TN
print("Akurasi =  " + str(
    ((true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)) * 100) + "%")
print("Presisi = " + str((true_positive / (true_positive + false_positive)) * 100) + "%")
print("Recall = " + str((true_positive / (true_positive + false_negative)) * 100) + "%")
print("DONE!")
