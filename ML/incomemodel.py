# incomemodel.py
# 의사결정나무 소득예측 모델

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree # 의사결정나무
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # 컨퓨젼매트릭스
import sklearn.metrics as metrics # 성능평가지표

# 데이터 임포트
df = pd.read_csv('assets/adult.csv')
#df.info()

# 데이터 전처리

# 연소득 5만달러 초과하면 high, 그렇지 않으면 low
df['income'] = np.where(df['income']=='>50K', 'high', 'low')
#print(df['income'].value_counts(normalize=True)) # 범주의 비율

# 타겟변수(income) 예측에 도움이 안되는 불필요한 변수 제거
df = df.drop(columns='fnlwgt') # 인구통계가중치 변수

# 원핫인코딩(one-hot encoding)을 이용한 문자타입변수를 숫자타입으로 변환
# 원핫인코딩 : 데이터를 1이나 0으로 표시
target = df['income'] # 1개 변수 (타겟변수=종속변수)
df = df.drop(columns='income') # 14개 변수 (예측변수=독립변수)
df = pd.get_dummies(df) # 원핫인코딩
df['income'] = target
#df.info(max_cols=np.inf) # 변수의 수와 관계없이 모든 변수의 정보 출력

# 훈련/테스트 데이터셋 분리
df_train, df_test = train_test_split(
    df,
    test_size=0.3,
    stratify=df['income'], # 범주별 비율을 통일할 변수
    random_state=1234
)
#print(df_train['income'].value_counts(normalize=True))
#print(df_test['income'].value_counts(normalize=True))

# 의사결정나무 모델

# 의사결정나무분류기 생성
clf = tree.DecisionTreeClassifier(
    random_state=1234,
    max_depth=3 # 트리의 최대 깊이
)

# 예측변수(독립변수), 타겟변수(종속변수) 추출
train_x = df_train.drop(columns='income')
train_y = df_train['income']

# 모델 학습
model = clf.fit(X=train_x, y=train_y)

# 시각화
# 그래프 설정
plt.rcParams.update({
    'figure.dpi': '100', # 그래프 해상도
    'figure.figsize': [12, 8] # 그래프 크기
})

# 의사결정나무 그래프
# tree.plot_tree(
#     model, # 모델
#     feature_names = train_x.columns, # 예측변수명들
#     class_names = ['high', 'low'], # 타겟변수 클래스명 (알파벳 오름차순)
#     proportion = True, # 클래스 배분 비율 표시 여부
#     filled = True, # 채움 여부
#     rounded = True, # 노드 테두리 둥글게 할지 여부
#     impurity = False, # 불순도 표시 여부
#     label = 'root', # 제목 표시 위치
#     fontsize = 10 # 글자 크기
# )
# plt.show()

# 예측을 위한 예측변수, 타겟변수 추출
test_x = df_test.drop(columns='income')
test_y = df_test['income']

# 예측
df_test['pred'] = model.predict(test_x)
#print(df_test)

# 예측 성능 평가

# 컨퓨젼매트릭스
conf_mat = confusion_matrix(
    y_true = df_test['income'], # 실제값
    y_pred = df_test['pred'], # 예측값
    labels = ['high', 'low'] # 레이블 (클래스 배치 순서, 문자 오름차순)
)
#print(conf_mat)

# 그래프 설정 초기화
plt.rcdefaults()

# 컨퓨젼매트릭스를 히트맵으로 표시
# p = ConfusionMatrixDisplay(
#     confusion_matrix = conf_mat, # 매트릭스 데이터
#     display_labels = ('high', 'low') # 타겟변수 클래스명
# )
# p.plot(cmap = 'Blues') # 컬러맵
# plt.show()

# 성능평가 지표

# 정확도
acc = metrics.accuracy_score(
    y_true = df_test['income'], # 실제값
    y_pred = df_test['pred'] # 예측값
)
print(acc)

# 정밀도
pre = metrics.precision_score(
    y_true = df_test['income'], # 실제값
    y_pred = df_test['pred'], # 예측값
    pos_label = 'high' # 관심 클래스
)
print(pre)

# 재현율
rec = metrics.recall_score(
    y_true = df_test['income'], # 실제값
    y_pred = df_test['pred'], # 예측값
    pos_label = 'high' # 관심 클래스
)
print(rec)

# f1-score
f1 = metrics.f1_score(
    y_true = df_test['income'], # 실제값
    y_pred = df_test['pred'], # 예측값
    pos_label = 'high' # 관심 클래스
)
print(f1)












































