# gridsearch.py
# 하이퍼파라미터튜닝 : GridSearch

# 라이브러리
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # GridSearch
from sklearn.svm import SVC #SVM
from sklearn.metrics import classification_report # 분류 보고서

# 데이터셋
iris = datasets.load_iris()

# X/y
X = iris.data
y = iris.target

# 훈련/테스트 분리
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=42)

# SVM
model = SVC()

# 하이퍼파라미터 후보 정의
# 3개 * 2개 * 3개 = 18개의 조합
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': [0.01, 0.1, 1]
}

# 그리드서치
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

# 출력
print('최적의 GridSearch 파라미터: ', grid.best_params_)
print('정확도: ', grid.score(X_test, y_test))
print(classification_report(y_test, grid.predict(X_test)))

























