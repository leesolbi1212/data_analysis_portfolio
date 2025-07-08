# dataexer1.py
# 데이터분석 실습 1

# 아래 순서대로 실습을 해봅시다!

# 필요한 라이브러리
import requests
import json
import pandas as pd

# 1. https://jsonplaceholder.typicode.com/todos
#    JSON데이터를 호출해 DataFrame을 생성
# get요청
response = requests.get('https://jsonplaceholder.typicode.com/todos')
# json변환
todosJson = response.json()
# DataFrame 생성
todosDF = pd.DataFrame(todosJson)
# print(todosDF)

# 2. userId가 5이상 인것들을 추출
subDF = todosDF.query('userId>=5')
#print(subDF)

# 3. userId 역순으로 정렬
sortedDF = subDF.sort_values("userId", ascending=False)
#print(sortedDF)

# 4. 결과를 result.csv로 저장
sortedDF.to_csv('result.csv')

