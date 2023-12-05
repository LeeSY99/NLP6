import pandas as pd
import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import tensorflow as tf
import numpy as np 
import keras_nlp
import keras_core as keras
import keras_core.backend as K

gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)

keras.utils.set_random_seed(42)

#훈련 데이터
train_data=pd.read_csv('/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv')
print(set(train_data['source']))

#키워드 별로 비율에 맞게 train set 구성
train=pd.concat([
    train_data[train_data.label==0].groupby('prompt_name',group_keys=False)
    .apply(lambda group: group.sample(frac=17497/len(train_data[train_data.label==0]), random_state=42)),
    train_data[train_data.label==1]
])

#변수설정
batch_size=6
num_folds=5
epochs=3

#Stratified K-Fold 교차 검증을 위해 계층화된 폴드를 생성
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
train = train.reset_index(drop=True) 
train['stratify'] = train.label.astype(str)+ train.source.astype(str)
train["fold"]=-1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['stratify'])):
    train.loc[val_idx, 'fold'] = fold
  train.groupby(["fold", "label","source"]).size()

import sentencepiece as spm
from keras_nlp.models import DebertaV3Preprocessor

# SentencePiece 모델 파일 경로 설정
proto = '/kaggle/input/d/leesyyyy/models2/vocab.spm'
tokenizer = keras_nlp.models.DebertaV3Tokenizer(proto)

# 전처리 클래스 생성
preprocessor = DebertaV3Preprocessor(
    tokenizer,
    sequence_length=200    # 최대 시퀀스 길이 (짧은 경우 패딩)
)

inp = preprocessor(train.text.iloc[0])  # 첫번째 행 텍스트 처리

# 처리된 출력 표시
for k, v in inp.items():
    print(k, ":", v.shape)
    
def preprocess_fn(text, label=None):
    text = preprocessor(text) 
    return (text, label) if label is not None else text

#tf.data.Dataset 구축
def build_dataset(texts, labels=None, batch_size=32,
                  cache=False, drop_remainder=True,
                  repeat=False, shuffle=1024):
    AUTO = tf.data.AUTOTUNE  # AUTOTUNE 옵션
    slices = (texts,) if labels is None else (texts, labels)  # 슬라이스 생성
    ds = tf.data.Dataset.from_tensor_slices(slices)  # 슬라이스에서 데이터셋 생성
    ds = ds.cache() if cache else ds  # 캐시 사용 여부에 따라 데이터셋 캐싱
    ds = ds.map(preprocess_fn, num_parallel_calls=AUTO)  # 전처리 함수 매핑
    ds = ds.repeat() if repeat else ds  # 반복 사용 여부에 따라 데이터셋 반복
    opt = tf.data.Options()  # 데이터셋 옵션 생성
    if shuffle: 
        ds = ds.shuffle(shuffle, seed=42)  # 셔플 사용 여부에 따라 데이터셋 셔플
        opt.experimental_deterministic = False
    ds = ds.with_options(opt)  # 데이터셋 옵션 설정
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)  # 데이터셋 배치
    ds = ds.prefetch(AUTO)  # 다음 배치를 미리 로드
    return ds  # 구축된 데이터셋 반환

def get_datasets(fold):
    # 데이터셋을 훈련 및 검증 세트로 분할
    train_df = train[train.fold != fold].sample(frac=1)
    
    # 지정된 폴드의 훈련 데이터 가져오기
    train_texts = train_df.text.tolist()
    train_labels = train_df.label.tolist()

    # 훈련 데이터셋 구축
    train_ds = build_dataset(train_texts, train_labels,
                             batch_size=batch_size, cache=False,
                             shuffle=True, drop_remainder=True, repeat=True)

    # 지정된 폴드의 검증 데이터 가져오기
    valid_df = train[train.fold == fold].sample(frac=1)
    valid_texts = valid_df.text.tolist()
    valid_labels = valid_df.label.tolist()

    # 검증 데이터셋 구축
    valid_ds = build_dataset(valid_texts, valid_labels,
                             batch_size=min(batch_size, len(valid_df)), cache=False,
                             shuffle=False, drop_remainder=True, repeat=False)

    return (train_ds, train_df), (valid_ds, valid_df)

def get_callbacks(fold):
    callbacks = []
    ckpt_cb = keras.callbacks.ModelCheckpoint(f'fold{fold}.keras',
                                              monitor='val_auc',
                                              save_best_only=True,
                                              save_weights_only=False,
                                              mode='max')  # 모델 체크포인트 콜백 획득
    callbacks.append(ckpt_cb)  # 체크포인트 콜백 추가        
    return callbacks

from keras_nlp.models import DebertaV3Classifier

def build_model():
    # DeBERTaV3Backbone 모델 생성
    backbone = keras_nlp.models.DebertaV3Backbone(
        vocabulary_size=128100,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=512,
    )
    
    # DeBERTaV3Classifier 모델 생성
    classifier = DebertaV3Classifier(
        backbone,
        preprocessor=None,
        num_classes=1 
    )
    inputs = classifier.input
    logits = classifier(inputs)

    # 최종 출력 계산
    outputs = keras.layers.Activation("sigmoid")(logits)
    model = keras.Model(inputs, outputs)

    # 옵티마이저, 손실 및 메트릭을 사용하여 모델 컴파일
    model.compile(
        optimizer=keras.optimizers.AdamW(5e-6),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=0.02),
        metrics=[
            keras.metrics.AUC(name="auc"),
        ],
        jit_compile=True
    )

    return model

model = build_model()
model.summary()

for fold in range(2, 3):
    (train_ds, train_df), (valid_ds, valid_df) = get_datasets(fold)
    callbacks = get_callbacks(fold)
    print('-' * 50)
    print(f'\tFold: {fold} | Model: deberta_v3_base_en\n\tBatch Size: {batch_size}')
    print(f'\tNum Train: {len(train_df)} | Num Valid: {len(valid_df)}')
    print('-' * 50)
    
    # TensorFlow 세션 초기화하고 전략 범위 내에서 모델 빌드
    K.clear_session()
    
    model = build_model()
        
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=callbacks,
        steps_per_epoch=int(len(train_df) / batch_size),
    )
    
    best_epoch = np.argmax(model.history.history['val_auc'])
    best_auc = model.history.history['val_auc'][best_epoch]
    best_loss = model.history.history['val_loss'][best_epoch]
    
     # 최적 결과를 출력하고 표시합니다
    print(f'\n{"=" * 17} FOLD {fold} RESULTS {"=" * 17}')
    print(f'>>>> BEST Loss  : {best_loss:.3f}\n>>>> BEST AUC   : {best_auc:.3f}\n>>>> BEST Epoch : {best_epoch}')
    print('=' * 50)

sub=pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
test=pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
predictions = model.predict(test_ds,batch_size=batch_size,verbose=1)
submission_df = pd.DataFrame({
    'id': test['id'],
    'generated': predictions.flatten()
})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)