# ex-stratifiedkfold.py
# stratified k-fold 실습 (~5:40)

# Stratified K-Fold 교차검증
# LogisticRegression(로지스틱회귀) 모델 사용
# 데이터 : 타이타닉 (train.csv)
# 폴드별 정확도, 폴드 평균 정확도 구하기
# 시각화 : 폴드별 클래스 분포 시각화
# Sex, Age를 features(X)로 사용
# Survived를 label(y)로 사용

# 1. 라이브러리 로딩
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. 데이터셋 로딩
df = pd.read_csv('assets/train.csv')

# 3. X, y 분리
df = df[["Sex", "Age", "Survived"]].dropna() # null 제거
df["Sex"] = df["Sex"].map({"male": 0, "female": 1}) # 숫자로 변경
X = df[["Sex", "Age"]] # features
y = df["Survived"] # label

# 4. StratifiedKFold 생성
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. 폴드별 정확도 저장 리스트
accuracies = []

# 6. 폴드별 정확도 연산 후 리스트에 저장 후 출력
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"폴드 {fold} 정확도: {acc:.4f}")

# 7. 폴드 평균 정확도 연산 후 출력
print(f"\n폴드 평균 정확도: {sum(accuracies)/len(accuracies):.4f}\n")

# 8. 폴드별 클래스 분포 시각화

fold_ratios = [] # 생존율 비율 저장용 리스트

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    y_test = y.iloc[test_idx]
    ratio = y_test.value_counts(normalize=True)
    print(f"Fold {fold} 생존율 비율: {dict(ratio)}")
    for cls, r in ratio.items():
        fold_ratios.append({'Fold': fold, 'Class': cls, 'Ratio': r})

# 시각화용 데이터프레임
ratio_df = pd.DataFrame(fold_ratios)

# 시각화
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(8, 5))
sns.barplot(data=ratio_df, x='Fold', y='Ratio', hue='Class')
plt.ylim(0, 1)
plt.title('StratifiedKFold 테스트셋 내 생존/사망 비율')
plt.ylabel('비율')
plt.xticks([0, 1, 2, 3, 4], ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
plt.legend(title='Class (0=사망, 1=생존)')
plt.tight_layout()
plt.show()





























