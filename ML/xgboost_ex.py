# xgboost_ex.py
# XGBoost 실습

# sklearn.datasets 내의 분류를 생성하는 make_classification함수를 활용하여
# 3개의 특성을 가진 이진 데이터셋을 만들고 XGBoost로 분류하고 시각화
# ~ 4시 20분

# 라이브러리 임포트
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==========================
# 1. 데이터 생성
# ==========================

# make_classification을 사용하여 이진 분류용 가상 데이터 생성
X, y = make_classification(
    n_samples=500,       # 데이터 샘플 수
    n_features=3,        # 총 특성 수 (3개의 입력 변수)
    n_informative=3,     # 유용한 정보가 담긴 특성 수 (모두 사용)
    n_redundant=0,       # 중복 특성 수 (없음)
    random_state=42      # 난수 시드 (결과 재현 가능)
)

# ==========================
# 2. 학습/테스트 데이터 분리
# ==========================

# train_test_split을 사용하여 학습용 80%, 테스트용 20%로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 3. DMatrix 변환 (XGBoost 형식)
# ==========================

# XGBoost는 자체적으로 최적화된 데이터 구조 DMatrix를 사용
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# ==========================
# 4. 하이퍼파라미터 설정
# ==========================

params = {
    'objective': 'binary:logistic',  # 이진 분류용 로지스틱 회귀
    'max_depth': 3,                  # 트리 최대 깊이 (복잡도 제어)
    'eta': 0.1,                      # 학습률 (0.1은 무난한 기본값)
    'eval_metric': 'logloss'        # 평가 지표: 로그 손실
}

# ==========================
# 5. 모델 학습
# ==========================

# XGBoost 모델 학습
bst = xgb.train(
    params=params,             # 하이퍼파라미터 설정
    dtrain=dtrain,             # 학습 데이터
    num_boost_round=100        # 트리 개수 (부스팅 반복 수)
)

# ==========================
# 6. 예측 및 평가
# ==========================

# 테스트 데이터에 대한 예측 수행 (확률값 반환)
pred_probs = bst.predict(dtest)

# 0.5 기준으로 이진 분류 결과 생성
pred_labels = (pred_probs > 0.5).astype(int)

# 정확도 출력
print("정확도:", accuracy_score(y_test, pred_labels))

# ==========================
# 7. 3D 시각화
# ==========================

# 테스트 데이터를 3차원 공간에 시각화
plt.rc('font', family='Malgun Gothic')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 산점도 그리기: 예측 클래스별로 색상 지정
ax.scatter(
    X_test[:, 0], X_test[:, 1], X_test[:, 2],    # X, Y, Z축 좌표
    c=pred_labels,                               # 색상은 예측된 클래스 값
    cmap='coolwarm',                             # 색상 맵
    edgecolors='k',                              # 점 테두리 색
    alpha=0.8                                     # 투명도
)

# 축 라벨 설정
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('XGBoost 분류 결과 (3D 시각화)')

plt.show()













