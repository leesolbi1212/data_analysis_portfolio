import csv
import re

# 텍스트 파일 읽기
with open('C:\AI_2504\wheel.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# 정규 표현식을 사용하여 데이터 추출
pattern = re.compile(r"\(\d+, '([^']+)', ([\d.]+), ([\d.]+), '([^']+)'\)")
matches = pattern.findall(data)

# CSV로 저장
with open('stations.csv', 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    # CSV 헤더 작성
    writer.writerow(['마커종류', '위도', '경도', '주소', '코멘트', '가중치'])

    # 데이터 작성
    for match in matches:
        station_name, lat, lng, comment = match
        writer.writerow([1, lat, lng, '', f"{station_name} {comment}", 0])

print("CSV 파일이 성공적으로 생성되었습니다: stations.csv")
