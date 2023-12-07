## 수많은 import
import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

## 대회에서 제공된 train, test data
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

## 데이터 증강을 통해 얻은 train data
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

## train 데이터 중복 제거
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

## tokenizer 관련 설정
LOWERCASE = False
VOCAB_SIZE = 30522

## tokenizer 인스턴스 설정
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel() 

## 특수토큰, trainer 설정
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

## 데이터셋 형태 맞추기
dataset = Dataset.from_pandas(test[['text']])

## 토크나이저 학습
def train_corp_iter(): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

## PreTrainedTokenizerFast사용으로 tokenizer설정
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

## testset 토큰화
tokenized_texts_test = []

for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

## trainset 토큰화
tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

## TfidfVectorizer 단어 벡터화
def dummy(text):
    return text

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

## 토큰화 된 텍스트들로 벡터화 학습
vectorizer.fit(tokenized_texts_test)

## Getting vocab
vocab = vectorizer.vocabulary_

## ngram 범위 (3, 5)로 설정
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer

## 메모리정리 (가비지 컬렉션)
gc.collect()

y_train = train['label'].values

## 모델 설정

## 테스트 데이터의 텍스트 개수가 5개 이하인 경우에는 아무 작업도 수행하지 않고, 그대로 제출 파일로 저장
if len(test.text.values) <= 5:
    sub.to_csv('submission.csv', index=False)

## 테스트 데이터의 텍스트 개수가 6개 이상
else:
		## 모델 설정
    clf = MultinomialNB(alpha=0.02)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
    p6={'n_iter': 1500,'verbose': -1,'objective': 'binary','metric': 'auc','learning_rate': 0.05073909898961407, 'colsample_bytree': 0.726023996436955, 'colsample_bynode': 0.5803681307354022, 'lambda_l1': 8.562963348932286, 'lambda_l2': 4.893256185259296, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
    lgb=LGBMClassifier(**p6)
    cat=CatBoostClassifier(iterations=1000,
                           verbose=0,
                           l2_leaf_reg=6.6591278779517808,
                           learning_rate=0.005689066836106983,
                           allow_const_label=True)

		## 가중치 설정
    weights = [0.1,0.3,0.3,0.3]
 
		## 앙상블 voting으로 모델 생성
    ensemble = VotingClassifier(estimators=[('mnb',clf),
                                            ('sgd', sgd_model),
                                            ('lgb',lgb), 
                                            ('cat', cat)
                                           ],
                                weights=weights, voting='soft', n_jobs=-1)
		## 학습
    ensemble.fit(tf_train, y_train)
    
		## 메모리정리 (가비지 컬렉션)
		gc.collect()
    
		## 최종 예측 결과 저장
		final_preds = ensemble.predict_proba(tf_test)[:,1]
    sub['generated'] = final_preds
    sub.to_csv('submission.csv', index=False)
    sub