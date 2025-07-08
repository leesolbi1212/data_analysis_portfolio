# boosting.py
# 부스팅

# 라이브러리
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# iris 데이터셋
iris = load_iris()
X = iris.data
y = iris.target

# train/test 분리
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoosting (Adaptive Boosting)
ada = AdaBoostClassifier(
    # 기본분류기 (기본값이 DecisionTreeClassifier), max_depth:트리 깊이
    estimator = DecisionTreeClassifier(max_depth=1), 
    # 부스팅 수행할 기본분류기 개수
    n_estimators = 50
)

# 모델 학습
ada.fit(X_train, y_train)

# 예측값
ada_pred = ada.predict(X_test)
#print(ada_pred)

# 정확도
ada_acc = accuracy_score(y_test, ada_pred)
#print(ada_acc)


## Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators = 365, # 부스팅을 수행할 분류기의 개수, 기본값 100
    # 학습률, 기본값 0.1, 학습률이 작을수록 모델이 안정적이고 과적합이 줄어듬
    learning_rate = 0.1,
    max_depth = 1 # 트리 최대 깊이, 기본값 3
)

# 모델 학습
gb.fit(X_train, y_train)

# 예측값
gb_pred = gb.predict(X_test)
#print(gb_pred)

# 정확도
gb_acc = accuracy_score(y_test, gb_pred)
#print('정확도: ', gb_acc)

# 트리 시각화

# 시각화 라이브러리
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd

# 데이터 로딩
iris = load_iris()
X = iris.data
y = iris.target

# train / test 분리
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터프레임
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#df.info()

# label 생성
df['label'] = [iris.target_names[x] for x in iris.target]

# 독립변수 / 종속변수
X = df.drop('label', axis=1)
y = df['label']

# 의사결정나무분류기 생성
clf = DecisionTreeClassifier(
    criterion = 'entropy',
    splitter = 'best',
    max_depth = 3,
    min_samples_leaf = 5
)

# 모델 학습
clf.fit(X, y)

# 예측값 출력
#print(clf.predict(X)[:3]) # setosa로 예측

# 독립변수 중요도 출력
# for i, column in enumerate(X.columns):
#     print(f'{column} 중요도 : {clf.feature_importances_[i]}')

# 시각화
# plt.figure(figsize=(12, 8))
# plot_tree(clf, feature_names=X.columns, class_names=clf.classes_)
# plt.show()

# clf 정보 확인 함수
import numpy as np
def get_info(dt_model, tree_type='clf'):
    tree = dt_model.tree_
    # 분할에 사용된 기준
    criterion = dt_model.get_params()['criterion']
    # 트리 유형이 유효한지 확인
    # assert : 주어진 조건이 참이 아니면 AssertionError를 발생시킴 (디버깅용으로 사용)
    assert tree_type in ['clf', 'reg']
    # 트리의 노드 수
    num_node = tree.node_count
    # 노드 정보를 저장할 리스트
    info = []
    # 트리의 각 노드 반복
    for i in range(num_node):
        # 각 노드의 정보를 저장할 딕셔너리
        temp_di = dict()
        # 현재 노드가 분할을 나타내는지 확인
        if tree.threshold[i] != -2: # -2 : leaf node
            # 분할에 사용된 특성과 임계값 저장
            split_feature = tree.feature[i]
            split_thres = tree.threshold[i]
            # 분할 질문
            temp_di['question'] = f'{split_feature} <= {split_thres:.3f}'
            # 불순도와 노드에 포함된 샘플 수
            impurity = tree.impurity[i]
            sample = tree.n_node_samples[i]
            # 불순도와 샘플 수 저장
            temp_di['impurity'] = f'{criterion} = {impurity:.3f}'
            temp_di['sample'] = sample
            # 예측된 값(회귀), 클래스 확률(분류)
            value = tree.value[i]
            temp_di['value'] = value
            # 분류 트리의 경우 예측된 클래스 레이블 저장
            if tree_type == 'clf':
                classes = dt_model.classes_
                idx = np.argmax(value)
                temp_di['class'] = classes[idx]
        info.append(temp_di)
    return info

# 함수 실행
#print(get_info(clf))

## Gradient Boosting 시각화
# gb = GradientBoostingClassifier(
#     n_estimators = 120,
#     learning_rate = 0.1,
#     max_depth = 1
# )
# gb.fit(X_train, y_train)
# first = gb.estimators_[0][0]
# plt.figure(figsize=(8, 5))
# plot_tree(first, filled=True, feature_names=iris.feature_names)
# plt.show()

## 회귀나무 (DecisionTreeRegressor)

# 보스턴 주택가격 데이터
from sklearn import datasets
boston = datasets.fetch_openml(
    'boston',
    version = 1,
    as_frame = True
)

# 데이터프레임
df = boston.frame

# 독립변수 / 종속변수 분리
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# DecitionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(
    criterion = 'squared_error',
    splitter = 'best',
    max_depth = 3,
    min_samples_leaf = 10,
    random_state = 100
)

# 학습
reg.fit(X, y)

# 예측값
print(reg.predict(X)[:3])

# 변수 중요도
for i, column in enumerate(X.columns):
    print(f'{column} 중요도 : {reg.feature_importances_[i]}')

# 시각화
plt.figure(figsize=(15, 12))
plot_tree(reg, feature_names=X.columns)
plt.show()







