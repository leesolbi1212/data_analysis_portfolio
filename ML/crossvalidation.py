# crossvalidation.py
# 교차 검증

# 라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

# 데이터프레임
df = pd.read_csv('assets/train.csv')

# 패쳐 / 레이블 분리
features = df.drop('Survived', axis=1)
labels = df['Survived']

# 데이터 전처리

# null 개수 확인
#print(features.isnull().sum())

# 결측치 다른값으로 대체
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)

# 불필요한 속성 제거
features = features.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['Sex'] = le.fit_transform(features['Sex'])
features['Embarked'] = le.fit_transform(features['Embarked'])
features['Cabin'] = le.fit_transform(features['Cabin'])

# 훈련세트 / 테스트세트 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    labels,
    test_size=0.2,
    random_state=53
)

# shape 확인
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# 의사결정트리
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(random_state=11)
dt.fit(X_train, y_train) # 모델 학습
pred = dt.predict(X_test) # 예측값
#print(f"정확도 : {accuracy_score(y_test, pred)}")

# 교차검증
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    dt, # 모델 (분류/예측)
    features, # 패쳐 (독립변수)
    labels, # 레이블 (종속변수)
    cv=10 # 폴드 수
)

# 교차검증시 각 검증의 정확도
# iter_count:반복회수(0부터 시작), accuracy:정확도
for iter_count, accuracy in enumerate(scores):
    print(f'{iter_count+1}번째 교차검증 정확도 : {accuracy}')

# 평균 정확도
print(np.mean(scores)) # 0.7744





























