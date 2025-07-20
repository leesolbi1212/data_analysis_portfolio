# softvoting.py
# 소프트 보팅 (실무에서는 소프트보팅을 많이 씀)

# 라이브러리
from sklearn.ensemble import VotingClassifier # 3개의 모델 앙상블
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# train / test 분리
X, y = load_iris(return_X_y=True) # 입력(X), 타겟(y)
# 입력 데이터 X (꽃받침, 꽃잎의 길이와 너비 등 4개 특성), 출력 타겟 y(꽃 종류 3가지 클래스)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=42)

# 개별 분류 모델 정의
lr = LogisticRegression() # 로지스틱 회귀 모델 : 선형적 경계를 찾아 데이터를 분류하는 간단하고 해석이 쉬운 모델
dt = DecisionTreeClassifier() # 결정 트리 모델 : 데이터를 분할 기준으로 계층적으로 분류하는 모델 (직관적, 성능 우수)
svm = SVC(probability=True) # SVM 모델 (확률 예측 가능하도록 설정) : 데이터를 나누는 최적의 경계를 찾는 모델 (비선형 문제에 강력한 성능)

# 정의한 3개의 분류기를 소프트보팅으로 결합한 Voting Classifier 생성
voting_clf = VotingClassifier(
    estimators = [('lr', lr), ('dt', dt), ('svm', svm)], # 각 모델에 이름 부여
    voting = 'soft' # 소프트보팅 사용
)

'''
모델1 (LogisticRegression): 클래스 A 70%, 클래스 B 30%
모델2 (DecisionTree): 클래스 A 60%, 클래스 B 40%
모델3 (SVC): 클래스 A 80%, 클래스 B 20%

→ 평균: 클래스A(70%), 클래스B(30%) → 최종 클래스는 클래스A
'''

# 분류기 학습 : 소프트 보팅 앙상블 모델에 학습 데이터를 제공하여 모델 학습 진행
voting_clf.fit(X_train, y_train)

# 모델 평가 / 정확도 출력
print('정확도: ', voting_clf.score(X_test, y_test))
