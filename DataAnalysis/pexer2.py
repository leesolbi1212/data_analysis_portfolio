# pexer2.py
# 지도시각화 실습
# 제주도 읍면동 경계 데이터 (assets/JEJU.geojson)
# h=> ttps://github.com/raqoon886/Local_HangJeongDong/tree/master
# 제주도 읍면동별 노인인구수 데이터 (assets/JEJU.csv)
# => https://www.data.go.kr/ (공공데이터포털에서 '제주특별자치도_읍면동노인인구현황' 검색)
# 제주도 읍면동별 ADM_CD 데이터 (ADM_CD.csv)

# 제주도 읍면동 경계 데이터 로딩
# 제주도 읍면동별 노인인구수 데이터 로딩
# 계급구간 설정
# 단계구분도 생성
# 브라우져에 지도시각화

import pandas as pd

# pycharm에서 웹브라우져를 통해 지도 표시
import webbrowser
def browser_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    webbrowser.open(html_page)

# 제주도 읍면동 경계 데이터
import json
geo_jeju = json.load(open("assets/JEJU.geojson", encoding="EUC-KR"))

# 제주도 읍면동별 노인인구수 데이터
older = pd.read_csv("assets/JEJU.csv", encoding="EUC-KR")

# ADM_CD를 int64타입에서 문자타입으로 변경
older["ADM_CD"] = older["ADM_CD"].astype(str)

# 계급구간 정하기
bins = list(older["노인인구수"].quantile([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]))

# 단계구분도 생성
import folium
map_jeju = folium.Map(
    location = [33.391281, 126.548625], # 중심위경도
    zoom_start = 11, # 확대레벨
    tiles = "cartodbpositron" # 지도유형
)
folium.Choropleth(
    geo_data = geo_jeju, # 지도 데이터
    data = older, # 노인인구수 데이터
    columns = ("ADM_CD", "노인인구수"), # 행정구역코드, 노인인구수
    key_on = "feature.properties.adm_cd", # 행정구역코드
    fill_color="Blues",
    nan_fill_color="White",
    fill_opercity=1,
    line_opercity=0.5,
    bins=bins
).add_to(map_jeju)

browser_open(map_jeju, "geo4.html")
