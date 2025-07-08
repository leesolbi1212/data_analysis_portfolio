# datedata.py
# 날짜데이터 라이브러리

# datetime
import datetime
today = datetime.date.today() # 오늘 날짜
print(today)
print(today.weekday()) # 요일의 숫자값
print(today + datetime.timedelta(days=100)) # 100일 후
print(today + datetime.timedelta(days=-100)) # 100일 전
print(today + datetime.timedelta(weeks=3)) # 3주 후
print(today + datetime.timedelta(hours=45)) # 45시간 후

day1 = datetime.date(2019, 1, 1)
day2 = datetime.date(2025, 4, 26)
print(day2 - day1) # 두 날짜의 차이

# calendar
import calendar
print(calendar.weekday(2025, 4, 26)) # 요일
print(calendar.isleap(2025)) # 윤년











