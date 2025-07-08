# ex-decisiontree.py
# 의사결정나무모델 실습 - 타이타닉 데이터셋
# (https://www.kaggle.com/competitions/titanic/data)
# train.csv 파일 사용 (assets/train.csv)
'''
| 컬럼명           | 데이터 유형     | 설명                                                              | 예시                        |
| ------------- | ---------- | --------------------------------------------------------------- | ------------------------- |
| `PassengerId` | int        | 탑승객 고유 ID (index 역할)                                            | 1, 2, 3 ...               |
| `Survived`    | int (0/1)  | 생존 여부 (종속변수/타겟) <br> `0 = 사망`, `1 = 생존`                         | 0, 1                      |
| `Pclass`      | int (1\~3) | 승객의 선실 등급 (사회경제적 지위 proxy) <br> `1 = 1등석`, `2 = 2등석`, `3 = 3등석` | 1, 3                      |
| `Name`        | string     | 승객 이름 (타이틀 추출에 유용)                                              | "Braund, Mr. Owen Harris" |
| `Sex`         | string     | 성별 (`male`, `female`)                                           | "male", "female"          |
| `Age`         | float      | 나이 (일부 결측치 존재)                                                  | 22.0, 38.0                |
| `SibSp`       | int        | 함께 탑승한 **형제/배우자 수**                                             | 1, 0                      |
| `Parch`       | int        | 함께 탑승한 **부모/자녀 수**                                              | 0, 1                      |
| `Ticket`      | string     | 티켓 번호 (불규칙적)                                                    | "A/5 21171"               |
| `Fare`        | float      | 승선 요금 (선실 등급과 상관관계 있음)                                          | 7.25, 71.2833             |
| `Cabin`       | string     | 선실 번호 (많은 결측치 존재)                                               | "C85", NaN                |
| `Embarked`    | string     | 탑승한 항구 <br>`C = Cherbourg`, `Q = Queenstown`, `S = Southampton` | "S", "C", "Q"             |

'''

# 실습 : 생존 여부를 예측

# 1. 라이브러리 임포트
import pandas as pd  # 데이터프레임 처리용
import numpy as np   # 수치 계산용
import matplotlib.pyplot as plt  # 시각화용
from sklearn.tree import DecisionTreeClassifier, plot_tree  # 결정트리 모델 및 시각화 도구
from sklearn.model_selection import train_test_split  # 데이터셋 분리용
from sklearn.metrics import (  # 모델 평가용 도구들
    accuracy_score,                # 정확도 계산
    classification_report,         # 정밀도, 재현율, F1-score 출력
    confusion_matrix,              # 혼동행렬 생성
    ConfusionMatrixDisplay         # 혼동행렬 시각화
)

# 2. 데이터 로드
df = pd.read_csv('assets/train.csv')

# 3. 데이터 확인
#df.info()
#print(df.head())

# 4. 필요한 열 선택 + 결측치 처리 (데이터 전처리)
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']] # 필요한 변수만 선택
df['Age'] = df['Age'].fillna(df['Age'].median()) # Age 없는 경우 중간값으로
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) # 성별을 숫자로

# 5. 입력(X), 출력(y) 나누기
X = df.drop('Survived', axis=1) # 입력값 : Pclass, Sex, Age, Fare
y = df['Survived'] # 출력값 : 생존여부 (0 또는 1)

# 6. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# 7. 모델 생성
clf = DecisionTreeClassifier(
    max_depth = 3, # 트리 최대 깊이
    random_state = 42
)

# 8. 모델 학습
clf.fit(X_train, y_train)

# 9. 예측 및 정확도 평가
y_pred = clf.predict(X_test) # 예측값
accuracy = accuracy_score(y_test, y_pred) # 정확도
print(f'정확도 : {accuracy:.2f}')

# 10. 의사결정나무 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names = X.columns, # 변수명
    class_names = ['Died', 'Survived'], # 클래스명 2개
    filled = True # 노드 색상으로 클래스 구분
)
plt.title('의사결정나무 시각화')
plt.show()

# 11. 모델 성능지표 출력 (정밀도, 재현율, f1-score)
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# 12. 컨퓨젼매트릭스 시각화
cm = confusion_matrix(y_test, y_pred) # 혼동행렬
disp = ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels = ['Died', 'Survived']
)
plt.figure(figsize=(6, 4))
disp.plot(cmap=plt.cm.Blues)
plt.title('혼동행렬(Confusion Matrix)')
plt.show()











