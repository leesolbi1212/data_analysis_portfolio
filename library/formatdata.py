# formatdata.py
# 형식화된 데이터 라이브러리

# 포맷팅(형식화된) 데이터 : 네트워크상에서 노드간에 주고 받는 형식이 정해진 데이터
# 데이터/파일 : 바이너리 (영상, 음성, 이미지 ....)
#               텍스트 (일반텍스트, CSV, JSON, XML, HTML, CSS, PY, JAVA ...)
# 텍스트에디터로 열었을때 글자가 깨지면 바이너리, 안깨지면 텍스트

# 네트워크 상의 주로 사용되는 데이터
'''
1. CSV (Comma Separated Value:콤마로 구분된 값)
ex) 1, 2, 3, 4, 5  또는  Hello, hi, Thanks
2. XML (Extensible Markup Language:확장가능한 표기 언어)
장점 : 데이터구조 + 데이터, 즉 데이터구조와 데이터를 모두 가지고 있음
단점 : 데이터 표현에 많은 바이트를 사용 (크기가 크다)
ex) <? xml version='1.0' encoding='utf-8'?>
    <persons>
        <person>
            <name>홍길동</name>
            <age>20</age>
        </person>
        <person>
            <name>강감찬</name>
            <age>30</age>
        </person>
    </persons>

3. JSON (JavaScript Object Notation:자바스크립트 객체 표기법)
장점 : 데이터 표현에 XML보다 적은 바이트를 사용
단점 : 데이터구조를 알 수 없음
ex) [
        {
            "name": "홍길동",
            "age": 20
        },
        {
            "name": "강감찬",
            "age": 30
        },
    ]

4. XML + JSON : 데이터구조도 알 수 있고, 데이터 바이트 수도 줄일 수 있음
ex) <? xml version="1.0" encoding="utf-8" ?>
    <persons>
        <person>
            {"name": "홍길동", "age": 20}
        </person>
        <person>
            {"name": "강감찬", "age": 30}
        </person>
    </persons>
'''

# CSV
import csv

# CSV파일에 쓰기
with open('csvdata.csv', mode='w', encoding='utf-8') as f:
    # writer(파일객체, 구분자, 따옴표종류)
    writer = csv.writer(f, delimiter=',', quotechar="'")
    writer.writerow(['홍길동', '30', '서울']) # 한 행의 데이터 쓰기
    writer.writerow(['강감찬', '40', '부산']) # 한 행의 데이터 쓰기

# CSV파일에서 읽기
with open('csvdata.csv', mode='r', encoding='utf-8') as f:
    print(f.read())

# XML
import xml.etree.ElementTree as ET
persons = ET.Element('persons')
person = ET.SubElement(persons, 'person')
name = ET.SubElement(person, 'name')
name.text = '홍길동'
age = ET.SubElement(person, 'age')
age.text = '20'
person = ET.SubElement(persons, 'person')
name = ET.SubElement(person, 'name')
name.text = '강감찬'
age = ET.SubElement(person, 'age')
age.text = '30'

# XML객체를 문자열로
xmlstr = ET.tostring(persons, encoding='utf-8').decode()
print(xmlstr)

# xmldata.xml파일에 XML문자열 쓰기
with open('xmldata.xml', mode='w', encoding='utf-8') as f:
    f.write(xmlstr)

# xmldata.xml파일을 읽어서 화면에 출력
with open('xmldata.xml', mode='r', encoding='utf-8') as f:
    print(f.read())

# JSON
import json

# JSON에서 문자열을 ""으로 묶어야 함!
json_data = '{"name": "홍길동", "age": 30, "city": "서울"}'

# JSON문자열을 파이썬 딕셔너리로 변환
data = json.loads(json_data)
print(data)

# 파이썬 딕셔너리를 JSON문자열로 변환
json_string = json.dumps(data)
print(json_string)

# JSON파일 읽기
with open('sample.json', 'r', encoding='utf-8') as f:
    print(json.load(f))

# JSON파일 쓰기
data = {
    "name": "홍길동",
    "age": 20,
    "city": "서울"
}
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# 중첩 JSON
json_data = '''
{
    "직원": {
        "name": "홍길동",
        "age": 20,
        "skills" : ["python", "ML", "DL"]
    }
}
'''
data = json.loads(json_data)
print(data)
print(data['직원']['name']) # 객체의 프라퍼티 키로 접근
print(data['직원']['skills'][1]) # 배열의 인덱스로 접근

# JSON 키 존재 여부
json_data = '{"name":"홍길동", "age":20}'
data = json.loads(json_data)
if 'city' in data: # city 키가 있다면
    print(data['city'])
else:
    print('city 키 없음!')

# JSON 키 정렬 후 출력
data = {
    "name": "홍길동",
    "age": 20,
    "city": "서울",
    "hobby": ["축구", "농구", "야구"]
}
json_string = json.dumps(data, indent=4, sort_keys=True)
print(json_string)

# 실습
# 아래 데이터를 JSON으로 변환해서 각 사람의 성적 총점을 출력
# 출력 형식 : 홍길동의 총점은 240
#            강감찬의 총점은 210
scores = [
    {"name": "홍길동","score": [90, 80, 70]},
    {"name": "강감찬","score": [80, 70, 60]}
]
# 1. scores를 JSON문자열로 변환
json_data = json.dumps(scores)
print(json_data)
# 2. JSON문자열을 파이썬 딕셔너리로 변환
data = json.loads(json_data)
print(data)
# 3. 총점 계산 후 출력
for person in data:
    print(f'{person["name"]}의 총점은 {sum(person["score"])}')











