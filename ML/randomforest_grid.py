# randomforest_grid.py
# 랜덤포레스트모델과 GridSearch를 활용한 실습 (2-1, 2-2 실습)
# kaggle의 mushrooms.csv
'''
타겟변수 : class 식용 여부 e:edible(식용), p:poisonous(독버섯)

1	cap-shape	버섯 갓의 형태	b: bell, c: conical, x: convex, f: flat, k: knobbed, s: sunken
2	cap-surface	갓의 표면 상태	f: fibrous, g: grooves, y: scaly, s: smooth
3	cap-color	갓의 색상	n: brown, b: buff, c: cinnamon, g: gray, r: green, p: pink, u: purple, e: red, w: white, y: yellow
4	bruises	눌림 자국 여부	t: bruises, f: no
5	odor	버섯의 냄새	a: almond, l: anise, c: creosote, y: fishy, f: foul, m: musty, n: none, p: pungent, s: spicy
6	gill-attachment	주름이 자루에 붙어 있는 방식	a: attached, d: descending, f: free, n: notched
7	gill-spacing	주름 간격	c: close, w: crowded, d: distant
8	gill-size	주름 크기	b: broad, n: narrow
9	gill-color	주름 색	k: black, n: brown, b: buff, h: chocolate, g: gray, r: green, o: orange, p: pink, u: purple, e: red, w: white, y: yellow
10	stalk-shape	버섯 자루의 형태	e: enlarging, t: tapering
11	stalk-root	자루의 뿌리 형태	b: bulbous, c: club, u: cup, e: equal, z: rhizomorphs, r: rooted, ?: missing
12	stalk-surface-above-ring	자루 위쪽 표면 상태	f: fibrous, y: scaly, k: silky, s: smooth
13	stalk-surface-below-ring	자루 아래쪽 표면 상태	f, y, k, s (위와 동일)
14	stalk-color-above-ring	자루 위쪽 색	n, b, c, g, o, p, e, w, y 등
15	stalk-color-below-ring	자루 아래쪽 색	동일
16	veil-type	덮개(막) 형태	p: partial (※ 모든 값이 p라 정보 없음)
17	veil-color	덮개의 색상	n: brown, o: orange, w: white, y: yellow
18	ring-number	고리 개수	n: none, o: one, t: two
19	ring-type	고리의 형태	c: cobwebby, e: evanescent, f: flaring, l: large, n: none, p: pendant, s: sheathing, z: zone
20	spore-print-color	포자 인쇄 색상	k: black, n: brown, b: buff, h: chocolate, r: green, o: orange, u: purple, w: white, y: yellow
21	population	개체수의 밀도	a: abundant, c: clustered, n: numerous, s: scattered, v: several, y: solitary
22	habitat	서식 환경	g: grasses, l: leaves, m: meadows, p: paths, u: urban, w: waste, d: woods
'''

# 라이브러리
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터셋 로딩 및 전처리
df = pd.read_csv('assets/mushrooms.csv')
#df.info()
df['target'] = df['class'].map({'e':0, 'p':1})
X = pd.get_dummies(df.drop(['class', 'target'], axis=1))
y = df['target']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2-1. 모델 생성
rf = RandomForestClassifier(random_state=42)

# 2-2. 모델 하이퍼파라미터 튜닝 (최적화 하이퍼파라미터 출력)
param_grid = {
    'n_estimators': [100, 200], # 사용할 트리의 개수
    'max_depth': [10, 20, None], # 각 트리의 최대 깊이
    'min_samples_split': [2, 5] # 노드 분할에 사용되는 최소 샘플 수
}
grid = GridSearchCV(
    rf, # 랜덤포레스트 모델
    param_grid, # 하이퍼파라미터
    cv = 5, # 교차검증 폴드 수
    scoring = 'accuracy', # 정확도를 기준으로 가장 좋은 모델 선택
    n_jobs = -1 # 모든 CPU코어를 활용하여 병렬 연산 수행 (학습 속도 향상을 위해)
)
grid.fit(X_train, y_train)
rf_best = grid.best_estimator_

# 3. 평가 및 시각화

# 테스트 데이터에 대해 최종 학습된 랜덤포레스트 모델로 예측
y_pred = rf_best.predict(X_test)  # 클래스 예측값 (0 or 1)

# 클래스 확률 중 '1' 클래스의 확률값 추출 (ROC-AUC 계산에 사용됨)
y_prob = rf_best.predict_proba(X_test)[:, 1]

# 분류 성능 리포트 출력 (정확도, 정밀도, 재현율, F1-score 등)
print(classification_report(y_test, y_pred))

# 혼동 행렬 출력 (TP, FP, FN, TN 구조로 분류 성능을 시각화)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC 스코어 계산 및 출력 (정밀도-재현율 trade-off 반영하는 종합 지표)
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve를 그리기 위한 FPR, TPR 좌표 계산
fpr, tpr, _ = roc_curve(y_test, y_prob)

# ROC Curve 시각화 시작
plt.figure()

# ROC 곡선 플롯 (TPR vs FPR)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")

# 기준선 (랜덤 모델) 추가: y=x 대각선
# plt.plot([0, 1], [0, 1], 'k--')
#
# # 플롯 제목, 축 레이블, 범례, 그리드 설정
# plt.title("Random Forest ROC Curve")
# plt.xlabel("FPR")          # False Positive Rate
# plt.ylabel("TPR")          # True Positive Rate (Recall)
# plt.legend()
# plt.grid()
# plt.show()                 # 그래프 출력

# Feature Importance 시각화

# 각 특성의 중요도 추출 (정보 획득량 기준)
importances = rf_best.feature_importances_

# 중요도 기준으로 상위 10개 피처의 인덱스 정렬 (내림차순)
indices = importances.argsort()[::-1][:10]

# 중요도 바 차트 시각화
sns.set_context("paper", font_scale=0.8) #폰트 스케일을 종이 출력 수준으로 낮추기
plt.figure(figsize=(6, 4))

# 상위 10개 특성에 대한 중요도 막대그래프 (X축: 중요도, Y축: 특성 이름)
sns.barplot(x=importances[indices], y=X.columns[indices])

# 차트 제목 추가
plt.title("Top 10 Feature Importance (RF)")
plt.xlabel("Importance", fontsize=10)
plt.ylabel("", fontsize=10)
plt.tight_layout() #여백 딱 맞게

# 차트 출력
plt.show()






































