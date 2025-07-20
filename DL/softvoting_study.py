# 1. 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.utils import resample
import numpy as np

# 2. 데이터 로드 및 분할
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. 교차검증 (Cross‑Validation)
# - 5‑Fold CV로 학습 데이터에서 모델 안정성 평가
'''
- 단일 테스트 셋 점수만 보면 과적합인지 모름 
- CV는 여러 번 평가해 평균 분산을 보므로, 모델 안정성을 객관적으로 판단할 수 있음
'''
clf = LogisticRegression(max_iter=200)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print("1) 교차검증 CV 점수:", cv_scores)
# 1) 교차검증 CV 점수: [1.         1.         0.86363636 1.         0.95454545]
# 5번 분할 중 세 번은 100%, 한 번은 85%, 한 번은 95%로 나옴 => 단일 분할 보다 평가가 불안정 할 수 있음을 보여줌

# 4. 러닝 커브 (Learning Curve)
# - 학습 곡선을 통해 데이터 규모 증가에 따른 오버/언더피팅 파악
# 데이터를 더 모으면 성능이 오를까? 지금 과적합인가?를 시각적으로 증명해보기 위함
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.2, 1.0, 5)
)
print("\n2) 러닝 커브 학습 크기:", train_sizes)
print("   러닝 커브 평균 학습 점수:", np.mean(train_scores, axis=1))
print("   러닝 커브 평균 검증 점수:", np.mean(test_scores, axis=1))
'''
2) 러닝 커브 학습 크기: [17 35 53 71 89]
   러닝 커브 평균 학습 점수: [1.         0.96571429 0.9509434  0.95774648 0.96629213]
   러닝 커브 평균 검증 점수: [0.93675889 0.91067194 0.93715415 0.94624506 0.96363636] 
'''
# 학습 정확도는 항상 95~100%로 매우 높고, 검증과 차이가 적어 과적합도 크지 않음.

# 5. 간단한 정규화·정칙화 비교 (Regularization Comparison)
# - 기본 C=1.0 vs C=0.1 비교
# 과적합이 의심될 때 모델을 더 단순하게 해서 일반화 성능을 확인해보고자 함.
clf_default = LogisticRegression(max_iter=200)
clf_reg = LogisticRegression(C=0.1, max_iter=200)
scores_default = cross_val_score(clf_default, X_train, y_train, cv=5)
scores_reg = cross_val_score(clf_reg, X_train, y_train, cv=5)
print("\n3) 기본 LR CV 점수:", scores_default)
print("   정칙화 LR(C=0.1) CV 점수:", scores_reg)
'''
3) 기본 LR CV 점수: [1.         1.         0.86363636 1.         0.95454545]
   정칙화 LR(C=0.1) CV 점수: [1.         0.95652174 0.81818182 0.95454545 0.90909091]
'''
# C=0.1(강한 정칙화)은 일부 Fold에서 정확도가 떨어졌지만, 모델 복잡도를 낮춰 과적합 위험을 완화

# 6. 단순 모델 대비 Baseline (Dummy Classifier)
# - 가장 빈도가 높은 클래스만 예측하는 더미 모델과 비교
# “우리 모델이 진짜 학습했는가?”를 확인하려면, 무작위·단순 전략보다 반드시 높은 성능을 보여야 하기 때문
dummy = DummyClassifier(strategy="most_frequent")
dummy_scores = cross_val_score(dummy, X_train, y_train, cv=5)
print("\n4) DummyClassifier CV 점수:", dummy_scores)
# 4) DummyClassifier CV 점수: [0.34782609 0.34782609 0.31818182 0.36363636 0.36363636]
# 학습 데이터에서 가장 많은 클래스(‘setosa’ 등)만 예측했을 때 약 34%의 정확도가 나옴

# 7. Bootstrap Resampling (OOB 평가)
# - 부트스트랩 샘플링 후 OOB(Out‑Of‑Bag) 점수 평균 계산
# 데이터 샘플링 변동성에 따른 성능 편차를 확인하고, 모델이 특정 샘플에 과도하게 의존하지 않는지 검증하기 위해
n_bootstraps = 50
n_samples = len(X_train)
boot_scores = []

for i in range(n_bootstraps):
    # 부트스트랩 샘플 인덱스와 OOB 인덱스 생성
    indices = resample(np.arange(n_samples), replace=True, random_state=i)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    if len(oob_indices) == 0:
        continue
    X_bs, y_bs = X_train[indices], y_train[indices]
    X_oob, y_oob = X_train[oob_indices], y_train[oob_indices]

    clf.fit(X_bs, y_bs)
    score = clf.score(X_oob, y_oob)
    boot_scores.append(score)

print("\n5) Bootstrap OOB 점수 예시 (상위 5개):", boot_scores[:5])
print("   Bootstrap 평균 OOB 점수:", np.mean(boot_scores))
'''
5) Bootstrap OOB 점수 예시 (상위 5개): [0.8636363636363636, 0.9142857142857143, 0.975, 0.9285714285714286, 0.9782608695652174]
   Bootstrap 평균 OOB 점수: 0.9540436934616691
'''
