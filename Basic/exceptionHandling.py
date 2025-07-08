# exceptionHandling.py
# 예외 처리
# 예외는 반드시 반드시 처리해야 함!!!

print('선 처리 중...')

# ZeroDivisionError: division by zero
#result = 10 / 0

# try: # 예외 발생 가능한 코드 블럭
#     ㅇ니ㅏ라어ㅣㄹㄴㅇㄹ
#     result = 10 / 0
# except NameError:
#     print('NameError 발생!')
# except ZeroDivisionError: # ZeroDivisionError 처리
#     print('ZeroDivisionError 발생!')
# finally: # 예외 발생여부와 관계없이 수행할 코드
#     print('예외 발생여부와 관계없이 수행할 코드')
#
# print('후 처리 중...')

# 사용자 정의 예외 (User Defined Exception)
# 개발자가 Exception을 상속받아 생성한 예외
# 예외처리 프로그래밍 기법 : 예외를 프로그래밍에 활용

class Under19Exception(Exception):
    def __str__(self):
        return '19세 이하 관람불가!'

class Under17Exception(Exception):
    def __str__(self):
        return '17세 이하 관람불가!'

age = 18
if age < 17:
    try:
        raise Under17Exception # 강제로 예외 발생
    except Under17Exception:
        print('Under17Exception 처리 함!')
    finally:
        print('예외발생 여부와 관계없이 수행할 코드블럭')
if age < 19:
    try:
        raise Under19Exception # 강제로 예외 발생
    except Under19Exception:
        print('Under19Exception 처리 함!')
    finally:
        print('예외발생 여부와 관계없이 수행할 코드블럭')








