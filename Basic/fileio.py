# fileio.py
# 콘솔입출력, 파일입출력
# 콘솔(console) : 표준입력장치(키보드) + 표준출력장치(모니터)

# 콘솔입력
#str = input('문자를 입력해 주세요!')

# 콘솔출력
#print(f'입력하신 문자는 {str}입니다!')

# num1 = input('첫번째 수를 입력하세요!')
# num2 = input('두번째 수를 입력하세요!')
# print('결과 : ', int(num1) + int(num2))

# 파일입출력
# 모드 : w 읽고쓰기, r 읽기, a 추가
# f = open('datafile.txt', 'a')
# f.write('Hello Python\n')
# f.close() # 메모리 자원 해제

f = open('datafile.txt', 'r', encoding='utf-8')
#print(f.readline()) # 한 줄 읽어서 문자열로
#print(f.read()) # 전체 줄 읽어서 문자열로
#print(f.readlines()) # 전체 줄 읽어서 리스트로
#f.close()

f = open('datafile2.txt', 'w', encoding='utf-8')
f.write('헬로우 파이썬!\n')
f.writelines(['헬로우\n', '파이썬\n'])
f.close()

# JSON : JavaScript Object Notation, 자바스크립트객체표기법
# 국제 표준 데이터 전송 포맷
# 자바스크립에서...
# [] : 배열(Array)
#      ex) [{}, {}, {}]
# {} : 객체(Object)
#      ex) {key:value, key:value, ...}
#      key와 문자열은 겹따옴표로 묶어줘야 함
# JSON파일은 일반 텍스트 파일이며 .json 확장자로 저장

# JSON객체 형태의 문자열
jsonStr = '{"name": "홍길동", "age": 20, "address": "서울"}'

f = open('jsonObj.json', 'w', encoding='utf-8')
f.write(jsonStr)

f = open('jsonObj.json', 'r', encoding='utf-8')
print(f.read())












