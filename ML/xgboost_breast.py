# xgboost_breast.py
# 유방암진단 데이터셋을 활용한 XGBOOST알고리즘

# 라이브러리
import pandas as pd
import numpy as np
import xgboost as xgb # XGBoost, 별도 설치 필요
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터셋 로딩
data = load_breast_cancer()

# 독립변수 / 종속변수
X = data.data
y = data.target

# 훈련 / 테스트 분리
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.2, random_state=42)

# DMatrix로 변환
# DMatrix : XGBoost에서 사용하는 데이터 형태
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 모델 파라미터 설정
# 하이퍼파라미터(Hyper Parameter) : 모델의 설정값
params = {
    # 목적함수 : 이진 분류용 로지스틱 회귀 (출력값은 확률)
    'objective': 'binary:logistic',
    # 각 트리의 최대 깊이 (값이 클수록 모델 복잡도 증가)
    'max_depth': 3,
    # 학습률(Learning rate), 작을수록 학습을 천천히 진행하므로 과적합 방지에 유리
    'eta': 0.1,
    # 평가 지표 : 로그 손실 (이진 분류의 확률 기반 손실 함수)
    'eval_metric': 'logloss'
}

# 모델 학습
bst = xgb.train(
    params, # 모델 학습에 제공할 하이퍼파라미터
    dtrain, # 학습데이터
    num_boost_round=100 # 부스팅 반복 회수 (=트리 개수)
)

# 예측
preds = bst.predict(dtest)

# 평가
pred_labels = (preds>0.5).astype(int)
print('정확도: ', accuracy_score(y_test, pred_labels))

# 시각화
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # 차원축소

# PCA를 통해서 고차원(30차원)을 2차원으로 축소
pca = PCA(n_components=2) # 2차원 축소용 PCA 생성
X_test_pca = pca.fit_transform(X_test) # 2차원으로 축소

# 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(8, 6))
plt.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    c=pred_labels,
    cmap='coolwarm',
    edgecolors='k',
    alpha=0.7
)
plt.title('XGBoost 예측 결과 (PCA 2D 시각화)')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.colorbar(label='예측 클래스')
plt.grid(True)
plt.show()






























