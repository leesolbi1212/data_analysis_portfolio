# softvoting.py
# 소프트 보팅

# 라이브러리
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# train / test 분리
X, y = load_iris(return_X_y=True) # 입력(X), 타겟(y)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25, random_state=42)

# 개별 분류 모델 정의
lr = LogisticRegression() # 로지스틱 회귀 모델
dt = DecisionTreeClassifier() # 결정 트리 모델
svm = SVC(probability=True) # SVM 모델 (확률 예측 가능하도록 설정)

# 정의한 3개의 분류기를 소프트보팅으로 결합한 Voting Classifier 생성
voting_clf = VotingClassifier(
    estimators = [('lr', lr), ('dt', dt), ('svm', svm)], # 각 모델에 이름 부여
    voting = 'soft' # 소프트보팅 사용
)

# 분류기 학습
voting_clf.fit(X_train, y_train)

# 모델 평가 / 정확도 출력
print('정확도: ', voting_clf.score(X_test, y_test))






























