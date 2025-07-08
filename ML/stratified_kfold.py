# stratified_kfold.py
# stratified k-fold (클래스 비율을 일정하게한 k-fold)

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer # 유방암 데이터셋
from sklearn.model_selection import StratifiedKFold # stratified k-fold
from sklearn.linear_model import LogisticRegression # 로지스틱회귀
from sklearn.metrics import accuracy_score # 정확도

# 데이터셋 로딩
data = load_breast_cancer()

# features / labels 분리
X = data.data
y = data.target # 0:악성, 1:양성

# stratified k-fold
skf = StratifiedKFold(
    n_splits=5, # 폴드 수
    shuffle=True, # 섞음
    random_state=42
)

# 정확도 저장 리스트
accuracies = []

# 교차검증 수행
for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    # 훈련/검증 데이터셋 분할
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # 로지스틱회귀 모델 생성 및 훈련
    model = LogisticRegression(max_iter=5000) # max_iter:최대 반복회수
    model.fit(X_train, y_train)
    # 예측값
    y_pred = model.predict(X_test)
    # 정확도 계산
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc) # 리스트에 정확도 추가
    # 폴드별 결과 출력 (폴드번호, 정확도 소수점4째자리)
    print(f'Fold {fold+1}, accuracy:{acc:.4f}')

# 평균 정확도를 소수점 4째자리까지 출력
print(f'\nAverage Accuracy : {np.mean(accuracies):.4f}')

# 시각화
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.plot(range(1, 6), accuracies, marker='o')
plt.title('폴드별 Stratified K-Fold 정확도')
plt.xlabel('폴드번호')
plt.ylabel('정확도')
plt.ylim(0.8, 1.0)
plt.grid(True)
plt.show()
































