# svm_iris.py
# iris데이터셋을 활용한 SVM알고리즘

# 라이브러리
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 데이터셋
iris = datasets.load_iris()
#print(iris)

# 독립변수 / 종속변수
#X = iris.data[:, :2] # 전체 행에서 2개 컬럼 (sepal_length, sepal_width)
X = iris.data[:, 2:4] # 전체 행에서 2개 컬럼 (petal_length, petal_width)
y = iris.target

# 훈련데이터셋 / 테스트데이터셋 분리
X_train, X_test, y_train, y_test \
    = train_test_split(X, y, test_size=0.3, random_state=42)

# SVM 분류기 생성
model = SVC(kernel='linear') # 기본커널 : rbf

# 모델 학습
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 정확도 출력
print('정확도: ', accuracy_score(y_test, y_pred))

# 시각화
def plot_decision_boundary(X, y, model):
    h = .02
    # 첫번째 컬럼의 최소값-1, 최대값+1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # 두번째 컬럼의 최소값-1, 최대값+1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # x축과 y축의 범위를 바탕으로 메쉬그리드 생성
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h),)
    # 예측을 위해 2차원 메쉬그리드를 1차원 배열로 변형한 후 모델 예측에 전달
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # 예측 결과를 다시 원래 메쉬그리드 모양으로 reshape
    Z = Z.reshape(xx.shape)
    # 결정 경계 영역을 색으로 시각화
    # contourf는 등고선의 내부를 채우는 함수
    plt.rc('font', family='Malgun Gothic')
    plt.contourf(xx, yy, Z, alpha=0.4)
    # 실제 데이터 포인트 산점도 시각화
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('꽃받침 길이')
    plt.ylabel('꽃받침 너비')
    plt.title('SVM결정 경계')
    plt.show()

plot_decision_boundary(X_train, y_train, model)

    


























