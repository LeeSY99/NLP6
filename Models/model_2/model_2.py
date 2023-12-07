!pip install -q language-tool-python --no-index --find-links ../input/daigt-misc/
!mkdir -p /root/.cache/language_tool_python/
!cp -r /kaggle/input/daigt-misc/lang57/LanguageTool-5.7 /root/.cache/language_tool_python/LanguageTool-5.7

import numpy as np
import pandas as pd
import regex as re
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import make_scorer, accuracy_score
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import language_tool_python
from concurrent.futures import ProcessPoolExecutor
from sklearn.naive_bayes import MultinomialNB
seed = 202

def seed_everything(seed=202):
    import random
    random.seed(seed)
    np.random.seed(seed)

seed_everything(seed)

tool = language_tool_python.LanguageTool('en-US')
def correct_sentence(sentence):
    return tool.correct(sentence)
def correct_df(df):
    with ProcessPoolExecutor() as executor:
        df['text'] = list(executor.map(correct_sentence, df['text']))

train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv")
external_train = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
external_train.rename(columns={'generated': 'label'}, inplace=True)

def how_many_typos(text):    
    return len(tool.check(text))

not_persuade_df = train[train['source'] != 'persuade_corpus']
persuade_df = train[train['source'] == 'persuade_corpus']
sampled_persuade_df = persuade_df.sample(n=6000, random_state=42)

all_human = set(list(''.join(sampled_persuade_df.text.to_list())))
other = set(list(''.join(not_persuade_df.text.to_list())))
chars_to_remove = ''.join([x for x in other if x not in all_human])

translation_table = str.maketrans('', '', chars_to_remove)
def remove_chars(s):
    return s.translate(translation_table)

train=pd.concat([train,external_train])
train['text'] = train['text'].apply(remove_chars)
train['text'] = train['text'].str.replace('\n', '')

test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
test['text'] = test['text'].str.replace('\n', '')
test['text'] = test['text'].apply(remove_chars)
correct_df(test)
df = pd.concat([train['text'], test['text']], axis=0)

vectorizer = TfidfVectorizer(ngram_range=(3, 5),tokenizer=lambda x: re.findall(r'[^\W]+', x), token_pattern=None, strip_accents='unicode')
vectorizer = vectorizer.fit(test['text'])
X = vectorizer.transform(df)

lr1 = LogisticRegression()
lr2 = LogisticRegression()
lr3 = LogisticRegression()

clf1 = MultinomialNB(alpha=0.02)
clf2 = MultinomialNB(alpha=0.02)
clf3 = MultinomialNB(alpha=0.02)

sgd_model1 = SGDClassifier(max_iter=8000, tol=1e-3, loss="modified_huber")
sgd_model2 = SGDClassifier(max_iter=10000, tol=5e-4, loss="modified_huber", class_weight="balanced") 
sgd_model3 = SGDClassifier(max_iter=15000, tol=3e-4, loss="modified_huber", early_stopping=True)

ensemble = VotingClassifier(
    estimators=[('lr1', lr1), ('lr2', lr2), ('lr3', lr3), ('mnb1', clf1), ('mnb2', clf2), ('mnb3', clf3),
                ('sgd1', sgd_model1), ('sgd2', sgd_model2), ('sgd3', sgd_model3)],
    weights=[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.15, 0.15],
    voting='soft'
)
ensemble.fit(X[:train.shape[0]], train.label)

preds_test = ensemble.predict_proba(X[train.shape[0]:])[:,1]

ntypos=test['text'].apply(lambda x: how_many_typos(x))
test['ntypos'] = -ntypos
test['generated'] = preds_test

submission = pd.DataFrame({
    'id': test["id"],
    'generated': test['generated']
})
submission.to_csv('submission.csv', index=False)