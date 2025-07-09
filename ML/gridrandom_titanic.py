# gridrandom_titanic.py
# 타이타닉 생존자 예측 모델 - GridSearchCV & RandomizedSearchCV 비교

# 0. 라이브러리
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
df = pd.read_csv('assets/train.csv')  # Kaggle Titanic 데이터 로드
df.info()

# 성별을 숫자로 변환 (male: 0, female: 1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# 나이 결측치를 중앙값으로 대체
df['Age'].fillna(df['Age'].median(), inplace=True)

# 승선항 결측치는 최빈값(S)으로 대체 후 숫자 매핑
df['Embarked'].fillna('S', inplace=True)
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# 사용할 피처 선택
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# 2. RandomForestClassifier 모델 학습 후 예측 및 정확도 출력
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))

# 3. GridSearchCV 하이퍼파라미터 튜닝 / 최적 하이퍼파라미터 및 정확도 출력
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 정확도 출력
print("Best GridSearch Params:", grid_search.best_params_)
print("Best GridSearch Accuracy:", grid_search.best_score_)

# 4. RandomizedSearchCV 하이퍼파라미터 튜닝 / 최적 하이퍼파라미터 및 정확도 출력
param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(4, 12),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 5)
}
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)
random_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 및 정확도 출력
print("Best RandomSearch Params:", random_search.best_params_)
print("Best RandomSearch Accuracy:", random_search.best_score_)

# 5. GridSearch vs RandomizedSearch 성능(정확도) 비교 막대그래프 시각화
scores = [
    accuracy_score(y_test, grid_search.best_estimator_.predict(X_test)),
    accuracy_score(y_test, random_search.best_estimator_.predict(X_test))
]
plt.rc('font', family='Malgun Gothic')
plt.bar(['GridSearch', 'RandomSearch'], scores)
plt.ylabel("테스트셋 정확도")
plt.title("하이퍼파라미터 튜닝 비교")
plt.ylim(0, 1)
plt.show()









































