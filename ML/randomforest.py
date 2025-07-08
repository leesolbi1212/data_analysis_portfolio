# randomforest.py
# 랜덤포레스트 : 여러개의 결정나무 생성하고 투표를 결정
#               > 과적합(한 쪽 클래스로 데이터가 몰리는 현상) 방지

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris # 붓꽃 데이터셋
from sklearn.ensemble import RandomForestClassifier # 랜덤포레스트
from sklearn.tree import plot_tree # 트리 시각화

# 데이터 셋
iris = load_iris()

# 독립/종속 변수
X = iris.data[:, 2:4] # 꽃잎길이(2번째컬럼), 꽃잎넓이(3번째컬럼)
y = iris.target # 품종 (0:setosa, 1:versicolor, 2:virginica)

# 모델 생성
rf = RandomForestClassifier(
    n_estimators = 3, # 결정나무 개수 3개
    max_depth = 3, # 각 결정나무의 최대 깊이 3개
    random_state = 42
)

# 모델 학습
rf.fit(X, y)

# 시각화
plt.figure(figsize=(20, 5)) # 20인치 넓이, 5인치 높이
for i in range(len(rf.estimators_)): # 트리 수만큼 반복
    plt.subplot(1, len(rf.estimators_), i+1) # 1행 n열 중 i+1번째 위치에 그림
    plot_tree(
        rf.estimators_[i], # i번째 결정나무
        feature_names = iris.feature_names[2:4], # 꽃잎길이, 꽃잎넓이
        class_names = iris.target_names, # 클래스명 3가지
        filled = True, # 노드를 색으로 채워 클래스 구분이 쉽도록
        rounded = True # 노드의 모서리를 둥글게
    )
    plt.title(f'Tree {i+1}') # 각 subplot 제목
plt.tight_layout() # 전체 그래프 크기를 내용에 맞춤
plt.show()


































