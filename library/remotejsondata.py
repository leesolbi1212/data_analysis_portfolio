# remotejsondata.py
# 원격지 JSON 데이터

# requests, json
import requests
import json

# requests 라이브러리를 통한 get요청
response = requests.get('https://jsonplaceholder.typicode.com/posts')
data = response.json()
print(data)

# requests 라이브러리를 통한 post요청
# 등록할 데이터 (payload)
sendData = {
    "userId": 1,
    "id": 101,
    "title": "sample title",
    "body": "sample body sample body sample body"
}
response = requests.post(
    'https://jsonplaceholder.typicode.com/posts',
    sendData
)
print(response.text)

# requests 라이브러리를 통한 put요청
# 등록할 데이터 (payload)
sendData = {
    "userId": 1,
    "id": 1,
    "title": "updated title",
    "body": "updated body updated body updated body"
}
response = requests.put(
    'https://jsonplaceholder.typicode.com/posts',
    sendData
)
print(response.text)

# requests 라이브러리를 통한 delete요청
response = requests.delete(
    'https://jsonplaceholder.typicode.com/posts/1'
)
print(response.text)

# urllib
from urllib.request import urlopen

# url에 연결
response = urlopen('https://jsonplaceholder.typicode.com/posts')
# 정상적으로 응답을 받으면
if response.getcode()==200: # 200:응답코드(status code) OK:응답텍스트(status text)
    # 응답데이터를 읽어서 utf-8 디코딩한 것을 json으로 변환
    data = json.loads(response.read().decode('utf-8'))
    # json 배열에서 객체 수만큼 반복
    for post in data:
        print(post['title']) # 제목 출력
else:
    print('에러!')
    
# 동기(synchronized)통신과 비동기(asynchronized)통신
# 동기 : 선행 실행결과가 후행 실행에 영향을 줄때 사용
#       방식 : 요청A > 응답A > 요청B > 응답B > 요청C ...
#       장점 : 선행 응답결과를 후행 요청에 사용할 수 있음
#       단점 : 동기 통신은 선행 응답후에 후행 요청을 해야하므로 속도가 느림
# 비동기 : 선행 실행결과가 후행 실행에 영향을 주지 않을때 사용
#       방식 : 요청A, 요청B, 요청C > 응답C, 응답A, 응답B ...
#       장점 : 비동기 통신은 요청 후 응답을 대기 하지 않으므로 속도가 빠름
#       단점 : 선행 응답결과를 후행 요청에 사용할 수 없음

# aiohttp : 비동기 통신에 사용되는 라이브러리
import aiohttp # http client 라이브러리
import asyncio # 비동기 통신 라이브러리

# 데이터 가져오는 비동기 함수
async def fetch_json(url):
    # 연결(Session) 생성
    async with aiohttp.ClientSession() as session:
        # 연결을 통해서 데이터를 가져옴
        async with session.get(url) as response:
            # await: 뒤의 처리가 완료될때까지 blocking(대기)
            data = await response.json()
            return data

# 비동기 함수를 호출하는 main함수
async def main():
    url = 'https://jsonplaceholder.typicode.com/posts'
    data = await fetch_json(url)
    print(json.dumps(data, indent=4)) # 들여쓰기 4칸

# main을 비동기로 호출
asyncio.run(main())

# 실습
# aiohttp라이브러리를 활용해서
# https://jsonplaceholder.typicode.com/users 데이터를 로딩한 후
# 사용자의 이름과 전화번호를 출력하는 프로그램 작성

async def main2():
    url = 'https://jsonplaceholder.typicode.com/users'
    response = await fetch_json(url)
    for user in response:
        print(user['name'], user['phone'])

asyncio.run(main2())

















