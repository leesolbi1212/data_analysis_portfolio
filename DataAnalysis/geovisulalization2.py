# geovisulalization2.py
# 서울시 동별 외국인 인구 단계 구분도 시각화

# 실습 (~ 5:40)
# 1. 서울시 동별 외국인 인구 수 단계 구분도 시각화
# 2. 구 경계를 추가

# 데이터
# 1. EMD_Seoul.geojson : 서울시 동 경계 지도 데이터
# 2. SIG_Seoul.geojson : 서울시 구 경계 지도 데이터
# 3. Foreigner_EMD_Seoul.csv : 서울시 동별 외국인 인구 데이터

import pandas as pd

# pycharm에서 웹브라우져를 통해 지도 표시
import webbrowser
def browser_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    webbrowser.open(html_page)

# 서울시 동 경계 지도 데이터
import json
geo_seoul = json.load(open("assets/EMD_Seoul.geojson", encoding="utf-8"))

# 서울시 동별 외국인 인구 데이터
foreigner = pd.read_csv("assets/Foreigner_EMD_Seoul.csv")

# code를 int64타입에서 문자타입으로 변경
foreigner["code"] = foreigner["code"].astype(str)

# 계급구간 정하기
bins = list(foreigner["pop"].quantile([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]))

# 단계구분도 생성
import folium
map_seoul = folium.Map(
    location = [37.56, 127],       # 중심위경도
    zoom_start = 12,               # 확대레벨
    tiles = "cartodbpositron"      # 지도유형
)
folium.Choropleth(
    geo_data = geo_seoul,       # 지도데이터
    data = foreigner,           # 통계데이터
    columns = ("code", "pop"),  # 행정구역코드, 인구
    key_on = "feature.properties.ADM_DR_CD", # 행정구역코드
    fill_color="Blues",
    nan_fill_color="White",
    fill_opercity=1,
    line_opercity=0.5,
    bins=bins
).add_to(map_seoul)

# 구 경계선 추가

# 구 경계 지도데이터
geo_seoul_sig = json.load(open("assets/SIG_Seoul.geojson", encoding="utf-8"))

folium.Choropleth(
    geo_data = geo_seoul_sig,       # 지도데이터
    fill_opacity=0,
    line_weight=4 # 선 굵기
).add_to(map_seoul)

browser_open(map_seoul, "geo3.html")




