# pca_xgboost_svm_ex.py
# PCA차원축소 후 XGBoost, SVM 알고리즘 활용 실습

# Digits 손글씨 숫자 데이터셋
# PCA로 2차원 축소
# 1. SVM 분류
# 2. XGBoost 분류
# 결과 : 정확도, 시각화

# 라이브러리
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
digits = load_digits()
X = digits.data  # (1797, 64) - 8x8 이미지의 픽셀
y = digits.target  # 0 ~ 9 정수 레이블

# 앞의 10개 샘플을 시각화
# plt.figure(figsize=(10, 2))
# for i in range(10):
#     plt.subplot(1, 10, i + 1)
#     plt.imshow(X[i].reshape(8, 8), cmap='gray')  # 8x8 이미지로 reshape
#     plt.title(f'{y[i]}', fontsize=12)
#     plt.axis('off')  # 축 제거
# plt.suptitle("Handwritten Digits (0-9)", fontsize=16)
# plt.tight_layout()
# plt.show()

# 2. PCA로 2차원으로 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. 데이터 분할 (훈련/테스트)
X_train, X_test, y_train, y_test \
    = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# 4. SVM 모델 학습 및 평가
svm_model = SVC(kernel='rbf', gamma='auto')  # RBF 커널 사용
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM 정확도:", acc_svm)

# 5. XGBoost 모델 학습 및 평가
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'multi:softmax',  # 다중 클래스 분류
    'num_class': 10,
    'eval_metric': 'mlogloss'
}
xgb_model = xgb.train(params, dtrain, num_boost_round=50)
y_pred_xgb = xgb_model.predict(dtest)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print("XGBoost 정확도:", acc_xgb)

# 6. 2D 시각화 (클래스 분포 확인용)
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.colorbar(label='Digit Class')
plt.title("Digits 데이터의 PCA 시각화 (2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()











