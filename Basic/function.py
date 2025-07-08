# function.py
# 함수

# 함수 정의
def add(a, b):
    return a + b

# 함수 호출
print(add(3, 5))

#print(add()) # TypeError
#print(add(3, 5, 7)) # TypeError

def multi(a, b=3): # 파라미터 기본값
    return a * b

print(multi(5, 5))
print(multi(5))

# 람다(lambda) 함수
# 일회성 함수로 이름없이 생성하고 사용하는 함수
# 문법 : def 생략, lambda 파라미터리스트: 코드블럭
# 람다함수는 값이므로 변수에 저장할수도 있고
# 파라미터에 인자로 전달할 수도 있음

lambda_add = lambda a, b: a + b
print(lambda_add(3, 3))

def calc(a, b, func):
    return func(a, b)

func1 = lambda a, b: a + b
print(calc(3, 5, func1))

func2 = lambda a, b: a * b
print(calc(3, 5, func2))










