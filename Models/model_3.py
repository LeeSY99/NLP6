import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

# 데이터 읽어오기
# 대회에서 제공된 train, test data
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
org_train = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/train_essays.csv')

# 데이터 증강을 통해 얻은 train data
train = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')

# train 데이터 중복 제거
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)

# tokenizer 관련 설정
LOWERCASE = False # 대소문자 설정
VOCAB_SIZE = 30522 # 어휘 크기 설정

# tokenizer 인스턴스 설정
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel() # byte 단위로 분할

# 특수토큰, trainer 설정
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

# 토크나이저에 적합한 형태로 데이터셋 생성
dataset = Dataset.from_pandas(test[['text']])

# 데이터셋을 반복문으로 토크나이져 학습
def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)

# PreTrainedTokenizerFast를 사용하여 토크나이저 설정
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# 테스트셋 텍스트 토큰화
tokenized_texts_test = []
for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

# 트레인셋 텍스트 토큰화
tokenized_texts_train = []
for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))

# TfidfVectorizer 단어 벡터화
def dummy(text):
    """
    이미 토큰화 한 텍스트를 그대로 백터화 하기 위해서 설정
    """
    return text

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

# 토큰화 된 텍스트들로 벡터화 학습
vectorizer.fit(tokenized_texts_test)

vocab = vectorizer.vocabulary_
# ngram 범위 (3, 5)로 설정
vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)


y_train = train['label'].values

# 모델 설정
bayes_model = MultinomialNB(alpha=0.02)
sgd_model = SGDClassifier(max_iter=10000, tol=1e-4, loss="modified_huber")

# 앙상블중 voting 방법으로 모델 생성, 가중치 설정
ensemble = VotingClassifier(estimators=[('sgd', sgd_model), 
                                        ('nb', bayes_model)],
                            weights=[0.85, 0.15], voting='soft', n_jobs=-1)
ensemble.fit(tf_train, y_train)

# 최종 예측 결과물
final_preds = ensemble.predict_proba(tf_test)[:,1]
sub['generated'] = final_preds
sub.to_csv('submission.csv', index=False)
sub