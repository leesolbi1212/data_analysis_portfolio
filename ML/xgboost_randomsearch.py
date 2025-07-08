# xgboost_randomsearch.py
# XGBoostClassifier모델과 RandomizedSearch 하이퍼파라미터 튜닝 실습

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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
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

# 2. XGBoost 모델
xgb = XGBClassifier(
    eval_metric='logloss', # 손실함수
    random_state=42
)

# 3. RandomizedSearchCV 실행 / 훈련
param_dist = {
    'n_estimators': [100, 200],            # 트리 개수 (많을수록 복잡한 모델)
    'learning_rate': [0.05, 0.1, 0.2],     # 학습률 (작을수록 천천히 학습)
    'max_depth': [3, 6, 10],               # 트리 최대 깊이 (깊을수록 복잡)
    'subsample': [0.8, 1.0]                # 데이터 샘플링 비율 (과적합 방지용)
}
rand = RandomizedSearchCV(
    xgb, # 모델
    param_distributions=param_dist, # 하이퍼파라미터
    n_iter=10, # 반복 회수
    cv=5, # 폴드 수
    # 성능평가 기준은 ROC-AUC 점수(모델의 이진분류 점수, 1에 가까울수록 완벽)
    scoring='roc_auc',
    n_jobs=-1, # 모든 CPU코어를 사용하여 병렬 처리
    random_state=42
)
rand.fit(X_train, y_train)

# 4. 최적 모델 추출
xgb_best = rand.best_estimator_

# 5. 예측 수행
y_pred = xgb_best.predict(X_test)
y_prob = xgb_best.predict_proba(X_test)[:,1]  # 클래스 1 확률

# 6. 평가지표
print(classification_report(y_test, y_pred))  # Precision, Recall, F1
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # 오차행렬
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 7. ROC Curve 시각화
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0,1],[0,1],'k--')  # 기준선
plt.title("XGBoost ROC Curve")
plt.xlabel("FPR");
plt.ylabel("TPR")
plt.legend();
plt.grid();
plt.show()

# 8. 특성 중요도 시각화 (상위 10개)
importances = xgb_best.feature_importances_
indices = importances.argsort()[::-1][:10]
plt.figure(figsize=(8,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Top 10 Feature Importance (XGBoost)")
plt.tight_layout()
plt.show()






















