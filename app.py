import pandas as pd
from sklearn.pipeline import Pipeline
from preprocessing import Preprocessing
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
import datetime
import time


nb_dataset = pd.read_csv("dataset2k.txt", sep="\t",
                         header=None, encoding='cp1252', error_bad_lines=False)
nb_dataset.columns = ['text', 'label']
positives = nb_dataset['label'][nb_dataset.label == 1]
negatives = nb_dataset['label'][nb_dataset.label == 0]
COLNAMES = ["id", "text"]
nltk.download('stopwords')


def word_count(text):
    return len(str(text).split())


nb_dataset["word_count"] = nb_dataset["text"].apply(word_count)
print("Dataset loaded successfully!")

all_words = []
for line in list(nb_dataset['text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())

dataset = nb_dataset

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
    nb_dataset['text'], nb_dataset['label'], test_size=0.2)
en_stopwords = set(stopwords.words("indonesian"))
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
vect = CountVectorizer(lowercase=True)
pipeline = Pipeline([('bow', vect),
                     ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])

parameters = {'bow__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (
    True, False), 'classifier__alpha': (1e-2, 1e-3)}

grid = GridSearchCV(pipeline, cv=10, param_grid={'bow__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'classifier__alpha': (1e-2, 1e-3)},
                    scoring="roc_auc",
                    verbose=1,
                    n_jobs=-1)
grid.fit(X_train, y_train)

means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']

joblib.dump(grid, "model.pkl")
# buat test model
model_NB = joblib.load("model.pkl")

y_preds = model_NB.predict(X_test)

print('akurasi dari train/test split: ',
      str(accuracy_score(y_test, y_preds) * 100) + "%")
print('confusion matrix: \n', confusion_matrix(y_test, y_preds))
print(classification_report(y_test, y_preds))

# testing
model_NB = joblib.load("model.pkl")


def label_to_str(x):
    if x == 0:
        return 'Negative'
    else:
        return 'Positive'


x = 0
text_ = [0] * len(nb_dataset)
label_ = [0] * len(nb_dataset)

for review in nb_dataset['text']:
    predict = model_NB.predict([review])
    text_[x] = review
    label_[x] = predict[0]
    x += 1

print("write ke csv")
hehe = {"text": text_, "label": label_}
hehe2 = pd.DataFrame(data=hehe)
hehe2.to_csv('test_ulang_dataset.csv', header=True, index=False,
             encoding='cp1252')
hasil_test_ulang = pd.read_csv(
    "test_ulang_dataset.csv", encoding='cp1252', header='infer', error_bad_lines=False)
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

    if nb_dataset['label'][i] == 1:
        if predicted_label == 1:
            true_positive += 1
        else:
            false_negative += 1

    if nb_dataset['label'][i] == 0:
        if predicted_label == 0:
            true_negative += 1
        else:
            false_positive += 1
    i += 1

    # print(nb_dataset['label'] == predicted_label)
    # print(predicted_label == 1)
Finish = time.time()
runningTime = Finish - Start
print("====== Hasil Sentimen Analisis NB ======")
now = datetime.datetime.now()
print("Run Pada : ")
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("True positive : " + str(true_positive))
print("True negative : " + str(true_negative))
print("False positive : " + str(false_positive))
print("False negative : " + str(false_negative))
print("Running Time --- %s seconds ---" % (runningTime))

# akurasi TP + TN / TP + FN + FP + TN
# menampilkan seluruh row (1000 row), berbeda dengan library yang hanya mengambil sample 20% dr total row
print("Akurasi =  " + str(
    ((true_positive + true_negative) / (true_positive + false_negative + false_positive + true_negative)) * 100) + "%")
print("Presisi = " + str((true_positive / (true_positive + false_positive)) * 100) + "%")
print("Recall = " + str((true_positive / (true_positive + false_negative)) * 100) + "%")
print("DONE!")
