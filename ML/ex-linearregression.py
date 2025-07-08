# ex-linearregression.py
# 선형회귀 실습
'''
실습 주제 : 선형회귀를 이용한 광고비 기반 매출 예측
실습 설명 : 회사는 다양한 광고 채널(예: TV, 라디오, 신문)에 광고비를 집행하고 있습니다.
          이때 광고비 지출이 매출(판매량)에 어떤 영향을 주는지 분석하고,
          광고비만 보고 매출을 예측할 수 있는 모델을 만들어봅니다.
          "모델 평가지표는 MSE, R^2으로 평가하고 시각화 해보기"
데이터셋 : Kaggle (https://www.kaggle.com/datasets/ashydv/advertising-dataset)
                 (Advertising.csv)
주요 컬럼:
- TV : TV 광고비
- Radio : 라디오 광고비
- Newspaper : 신문 광고비
- Sales: 실제 판매량
'''

# -----------------------------------------------
# 1. 필요한 라이브러리 불러오기
# -----------------------------------------------
import pandas as pd                                # 데이터프레임 처리
import numpy as np                                 # 수치 계산
import matplotlib.pyplot as plt                    # 시각화 도구
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.metrics import mean_squared_error, r2_score  # 평가 지표



