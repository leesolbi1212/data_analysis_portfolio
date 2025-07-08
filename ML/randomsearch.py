# randomsearch.py
# 하이퍼파라미터튜닝 : RandomSearch

# 라이브러리
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV # RandomizedSearch
from sklearn.svm import SVC #SVM
from scipy.stats import uniform

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
param_search = {
    'C': uniform(0.1, 10), # 0.1 ~ 10 범위의 어떤 수
    'kernel': ['linear', 'rbf'],
    'gamma': uniform(0.001, 1) # 0.001 ~ 1 범위의 어떤 수
}

# 랜덤서치
random_search = RandomizedSearchCV(
    model,
    param_distributions = param_search,
    n_iter = 10, # 반복회수
    cv = 5, # 폴드 수 (K-Fold)
    random_state = 50
)
random_search.fit(X_train, y_train)

# 출력
print('RandomSearch 파라미터: ', random_search.best_params_)
print('정확도: ', random_search.score(X_test, y_test))

























