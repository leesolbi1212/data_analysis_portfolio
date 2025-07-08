# stringdata.py
# 문자열 처리 라이브러리

## textwrap 라이브러리
import textwrap

str = 'Hello Python'

# 문자열 길이 축약
# textwrap.shorten(문자열, 길이, 대체문자열)
print(textwrap.shorten(str, width=10, placeholder='...'))

# 문자열 공백기준으로 리스트로 변환
# textwrap.wrap(문자열, 길이)
wrapstr = textwrap.wrap(str, width=11)
print(wrapstr)

## re 라이브러리 (Regular Expression, 정규표현식)
# 정해진 패턴문자열들을 사용해서 문자열내의 문자열을
# 검색, 치환하는데 사용되는 표현식

# 자주 사용되는 패턴 문자들
# ^ : 문자열의 시작
# [^문자열] : 문자열이 아닌 문자
# $ : 문자열의 끝
# . : 임의의 한 문자
# * : 문자가 0개 이상
# + : 문자가 1개 이상
# ? : 문자가 0개 또는 1개
# [] : 문자의 집합 범위
# {} : 문자열의 반복 회수
# | : OR
# \b : 단어의 경계
# \B : 단어의 경계가 아닌 문자
# \s : 공백문자
# \S : 공백문자가 아닌 문자
# \w : 워드 (영문대소문자, 숫자, _)
# \W : 워드가 아닌 문자
# \d : 숫자
# \D : 숫자가 아닌 문자
# \\ : \문자 자체

# 플래그 (flag)
# 패턴문자열 뒤에서 검색에 대한 옵션을 지정하는 문자
# 하나를 사용하거나 조합해서 사용 가능
# i (ignorecase) : 대소문자 구별 없음
# g (global) : 전역에서 탐색
# m (multiline) : 여러줄에서 시작과 끝 탐색

import re

str = "홍길동의 전화번호는 010-1234-5678"
# 숫자3개-숫자4개-숫자4개가 나오는 정규표현식 패턴
# compile : 패턴 생성
pattern = re.compile("(\d{3})-(\d{4})-(\d{4})")
# 검색 순서에 따라 g<검색순서> 형태로 사용
# sub : 문자열 대체
print(pattern.sub(r'(\g<1>)\g<2>-\g<3>', str))

text = 'I like apple pie'
# findall : 문자열에서 패턴에 해당하는 문자열을 리스트로 반환
result = re.findall(r'apple', text)
print(result)

# 1개 이상의 숫자를 검색
text = 'My phone number is 010-1234-5678'
result = re.findall(r'\d+', text)
print(result)

# 영문대문자 검색
text = 'Hello Python'
result = re.findall(r'[A-Z]', text)
print(result)

# 영문소문자 1개이상 검색
result = re.findall(r'[a-z]+', text)
print(result)

# 문자열내의 불필요한 공백 제거
text = 'Hello     Python     This    is   me'
result = re.sub(r'\s+', '', text)
print(result)

# 문자열 시작과 끝 검색
text = 'Hello World'
result = re.findall(r'^Hello|World$', text)
print(result)

# 날짜형식 검색
text = '오늘은 2025-4-26일이고 내일은 2025-4-27일 입니다'
result = re.findall(r'\d{4}-\d{1}-\d{2}', text)
print(result)
text = '오늘은 2025-4-26일이고 내일은 2025-04-27일 입니다'
result = re.findall(r'\d{4}-\d{1,2}-\d{2}', text)
print(result)

# 문자열에서 URL 검색
text = '홈페이지 주소는 http://myhome.com 또는 http://www.myhome.com'
result = re.findall(r'http://[^\s]+', text)
print(result)


# 실습 1
# 주민등록번호 패턴 (간소화 버젼)
# 숫자6개-숫자7개, 뒷숫자 맨 앞자리는 1~4중 하나
text = '990101-1234567 050202-4123456'
result = re.findall(r'\d{6}-[1-4]\d{6}', text)
print(result)

# 실습 2
# 이메일 패턴 (간소화 버젼)
# 영문또는숫자가3~12개@영문또는숫자가3~12개 나오고 .com 또는 .kr 또는 .co.kr
# ?: : 비캡춰그룹
text = 'abc123@mycompany.com 123abc@mycompany.co.kr'
result = re.findall(r'[a-zA-Z0-9]{3,12}@[a-zA-Z0-9]{3,12}\.(?:com|kr|co\.kr)', text)
print(result)

# i 플래그 : 대소문자 구별없이 탐색
text = 'Hello hello HeLLo hElLo'
pattern = re.compile(r'hello', re.I)
result = pattern.findall(text)
print(result)

# g 플래그 : 전역 탐색
# 파이썬은 기본적으로 전역 검색을 함
text = 'cat dog cat bird cat'
result = re.findall(r'cat', text)
print(result)

# m 플래그 : 멀티라인 탐색
text = '''Hello World
Hello Python
Hello AI'''
pattern = re.compile(r'^Hello', re.M)
result = pattern.findall(text)
print(result)












