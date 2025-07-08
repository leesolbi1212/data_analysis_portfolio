# logic.py
# 로직 (분기, 반복)

def p(value):
    print(value, '\n')

# 분기 (if)
a = 10
if a>5:
    p('a는 5보다 큽니다!')
else:
    p('a는 5보다 크지 않습니다!')

a = 3
if a>10:
    p('a는 10보다 큽니다!')
elif a>5:
    p('a는 10보다 크지는 않지만 5보다는 큽니다!')
else:
    p('a는 5보다 작습니다!')

# 반복

# for

# 리스트의 요소수만큼 반복
for i in [1, 2, 3, 4, 5]:
    print(i, end=' ') # i 출력한 후에 공백문자 추가

print()

# 문자열의 문자수만큼 반복
for i in 'Hello':
    print(i, end=' ')

print()

# range함수 : 범위를 지정
# range(시작, 종료+1, 스텝)
for i in range(1, 10): # 1~9
    print(i, end=' ')

print()
    
for i in range(1, 101, 2): # 1~100까지 스텝2로 반복
    print(i, end=' ')

print()

# 중첩 for
for i in range(2, 10): # 2~9
    for j in range(1, 10): # 1~9
        print(f'{i}*{j} = {i*j}', end=' ')
    print()

print()

# 별찍기 1번
for i in range(1, 6): # 1~5, 줄 수
    for j in range(1, i+1): # 1~i+1, 별 수
        print('*', end='')
    print()

print()

# 별찍기 2번
for i in range(1, 6): # 1~5, 줄 수
    for j in range(1, 7-i): # 1~7-i, 별 수
        print('*', end='')
    print()

print()

# 딕셔너리 반복
dic = {'a':1, 'b':2, 'c':3}
for i in dic.keys():
    p(i)
for i in dic.values():
    p(i)
for i in dic.items():
    p(i)

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for i in nums:
    if i % 2 == 0: # 짝수
        # 아래 코드 수행하지 않고 다음번 반복 수행
        continue
    if i==9:
        # 가장 가까운 반복문을 즉시 탈출
        break
    else:
        p(i)

# while
# 조건식이 True인동안 반복
# 언젠가는 조건이 False인 경우가 있어야 함

# 무한반복 조심!
#while True:
#    p('hello')

a = 0
while a < 10:
    p(a)
    a = a + 1

a = 0
while a < 10:
    if a==5:
        break
    p(a)
    a = a + 1












