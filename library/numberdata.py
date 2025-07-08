# numberdata.py
# 숫자 데이터 라이브러리

# math : 수학관련 라이브러리
import math

print(math.gcd(60, 80, 100)) # 최대공약수
print(math.lcm(15, 25)) # 최소공배수

# decimal : 소수관련 라이브러리
from decimal import Decimal
print(0.1 * 3)
print(Decimal('0.1') * 3)

# fractions : 분수관련 라이브러리
from fractions import Fraction
print(Fraction(1.5))

# random : 랜덤수관련 라이브러리
import random
print(random.random()) # 0.0보다 크거나 같고 1.0보다 작은 실수
print(random.randint(1, 45)) # 1~45중 임의의 정수

# 로또 숫자 6개 추출
lottoNum = set() # 중복없는 데이터 저장을 위해 set 사용
while True:
    lottoNum.add(random.randint(1, 45))
    if (len(lottoNum)==6):
        break
print(list(lottoNum))

# statistics : 기초통계 (최대, 최소, 평균, 25%, 50%(중앙값), 75%)
import statistics
score = [38, 54, 45, 87, 92, 66, 28, 99]
print(statistics.mean(score)) # 평균
print(statistics.median(score)) # 중앙값
print(statistics.stdev(score)) # 표준편차
print(statistics.variance(score)) # 분산

import numpy as np
print(np.percentile(score, 25)) # 25%
print(np.percentile(score, 75)) # 75%
print(np.max(score)) # 최대
print(np.min(score)) # 최소










