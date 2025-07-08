# dataTypes.py
# 데이터타입

# 출력 함수
def p(value):
    print(value, '\n')

# 숫자타입
p(1) # 정수
p(0.1) # 실수 (부동소수점 수)
p(3E5) # 3곱하기 10의 5승
p(0o10) # 8진수
p(0xFF) # 16진수
p(1+2) # 덧셈
p(3**3) # 3의 3승
p(7%4) # 7을 4로 나눈 나머지
p(7//4) # 7을 4로 나눈 몫

# 문자열타입
p("Hello")
p('Hello')
# 여러줄 문자열
p('''
hello
hello
hello
''')
p("Hello 'Tom'") # 중첩 지켜줘야 함
p('Hello "Tom"')
p('Hello\nTom') # 줄바꿈 문자
p('Hello'*3) # 문자열 반복
p(len('Hello')) # 문자열 길이

str = 'Hello There'
p(str[0]) # 인덱스0 문자
p(str[:3]) # 인덱스 0~2
p(str[3:]) # 인덱스 3부터 끝까지
p(str[3:5]) # 인덱스 3~4
p(str[::2]) # 0부터 끝까지 스텝 2

# 동적타이핑 (Dynamic Typing)
# 인터프리터 언어는 코드 실행시 변수에 값이 할당될 때
# 변수의 타입이 정해짐
# 변수가 가지는 값의 타입이 변경되면 변수의 타입이 변경됨
str = 'jane' # 실행하면 str의 타입은 문자열타입
str = 100 # 실행하면 str의 타입은 숫자타입

# 출력 형식
name = '홍길동'
age = 20
p('%s은 %d살 입니다!' %(name, age)) # 변수 할당
p(f'{name}은 {age}살 입니다!') # 변수 할당
p('%10s' %'Hello') # 출력 자리수 지정
p('%10.4f' %3.141592) # 전체 10자리 중 소수점이하 4자리

# 문자열 처리 함수
str = 'hello there'
p(str.count('e')) # e문자의 수
p(str.find('e')) # e가 처음 나온 곳의 인덱스
p(str.find('p')) # 없으면 -1
p(str.upper()) # 대문자로 변경
p(str.lower()) # 소문자로 변경
p(str.replace('e', 'k')) # e를 k로 대체
p(str.split()) # 공백문자 기준으로 문자열을 잘라서 리스트 생성
p(str.split('e')) # e문자 기준으로 문자열을 잘라서 리스트 생성
p('010-1234-5678'.split('-')) # -문자 기준으로 ...

# 리스트 (list, [])
even = [2, 4, 6, 8] # length:4, index범위:0~3
odd = [1, 3, 5, 7]
print(even, odd)

p(even[0]) # 인덱스0 값
p(even[0:]) # 처음부터 끝까지
p(even[:2]) # 0~1까지
p(even[::2]) # 처음부터 끝까지 스텝 2
p(even[-1]) # 뒤에서 첫번째

p(even + odd) # 리스트 병합
p(even * 3) # 리스트 3번 반복

p(even.index(6)) # 6이 처음 나온 곳의 인덱스
p(len(even)) # 요소의 수

del even[0]
p(even)

even.append(10) # 맨 뒤에 10을 추가
p(even)

even.reverse() # 요소 순서 반전
p(even)

even.sort() # 오름차순 정렬
p(even)

even.insert(0, 2) # 0인덱스에 2를 삽입
p(even)

even.remove(2) # 2요소 제거
p(even)

even.pop() # 맨 뒤 요소 제거
p(even)

even.extend([10, 12]) # 리스트에서 요소를 꺼내 기존리스트에 추가
p(even)

even.append([10, 12]) # 리스트 자체에 기존리스트에 추가
p(even)

# 튜플 (Tuple, ())
yoil = ('월', '화', '수', '목', '금', '토', '일')
p(yoil)

# yoil[0] = 'monday' # 튜플은 값 변경 불가
# p(yoil)

# 딕셔너리 (Dictionary, {})
# 딕셔너리는 아이템(키:값)들의 모음
dic = {
    'name': '홍길동',
    'age': 20,
    'address': '서울'
}
p(dic)

p(dic['name']) # name키의 값
p(dic.get('name'))

dic['name'] = '이순신' # name키의 값 변경
p(dic)

dic['gender'] = '남' # 아이템 추가
p(dic)

del dic['gender']
p(dic)

# 딕셔너리 함수
p(dic.keys()) # dict_keys타입
p(list(dic.keys())) # 리스트로 변경
p(dic.values()) # dict_values타입
p(list(dic.values())) # 리스트로 변경
p(dic.items()) # dict_items타입
p(list(dic.items())) # 리스트로 변경
p('name' in dic) # 딕셔너리에 name키가 있는지
p('gender' in dic)

# 집합 (Set, {})
# 중복된 요소는 하나만 저장
s = {1, 2, 3, 4, 5, 1, 2, 3}
p(s)

# 리스트로 set 생성
s = set([1, 2, 3, 4 ,5, 1, 2, 3])
p(s)

# 집합 연산
s1 = {1, 2, 3, 4, 5}
s2 = {3, 4, 5, 6, 7}
p(s1 & s2) # 교집합
p(s1 | s2) # 합집합
p(s1 - s2) # 차집합
p(s2 - s1) # 차집합

# 불리언 (Boolean)
# True 또는 False의 값만 저장
p(1==1) # T
p(2<1) # F
p(bool('hello')) # T
p(bool('')) # F
p(bool([1, 2, 3])) # T
p(bool([])) # F
p(bool({'name':'홍길동'})) # T
p(bool({})) # F
p(bool(0)) # T
p(bool(100)) # F
p(1=='1') # F





