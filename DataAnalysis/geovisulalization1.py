# geovisulalization1.py
# 대한민국 시군구별 인구데이터 지도시각화

# pycharm에서 웹브라우져를 통해 지도 표시 함수
import webbrowser
def browser_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    webbrowser.open(html_page)

# 대한민국 시군구 경계 좌표 데이터
import json
geo = json.load(open('assets/SIG.geojson', encoding="utf-8"))
#print(geo)
#print(type(geo)) # dictionary

# 행정구역 코드
#print(geo['features'][0]['properties']['SIG_CD'])

# 위도, 경도 좌표
#print(geo['features'][0]['geometry']['coordinates'])

# 시군구별 인구데이터
import pandas as pd
df_pop = pd.read_csv('assets/Population_SIG.csv')
#df_pop.info()

# code를 int64타입에서 문자타입으로 변환
df_pop['code'] = df_pop['code'].astype(str)

# 단계구분도 생성
import folium
# 지도 형태
map_sig = folium.Map(
    location = [35.95, 127.7], # 중심 위/경도
    zoom_start = 8, # 확대 레벨
    tiles = 'cartodbpositron' # 지도의 타일
)
# 지도 데이터
folium.Choropleth(
    geo_data = geo, # 지도데이터
    data = df_pop, # 인구데이터
    columns = ('code', 'pop'), # 행정구역코드, 인구수
    key_on = 'feature.properties.SIG_CD' # geo 행정구역코드를 키로 사용
).add_to(map_sig) # 지도에 데이터 추가

# 지도를 웹브라우져에 표시
browser_open(map_sig, 'geo1.html')

# 계급구간 정하기
bins = list(df_pop['pop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))
#print(bins)

folium.Choropleth(
    geo_data = geo, # 지도데이터
    data = df_pop, # 인구데이터
    columns = ('code', 'pop'), # 행정구역코드, 인구수
    key_on = 'feature.properties.SIG_CD', # geo 행정구역코드를 키로 사용
    fill_color = 'YlGnBu', # 채움 색상
    fill_opacity = 0.5, # 투명도
    bins = bins # 계급구간
).add_to(map_sig) # 지도에 데이터 추가

browser_open(map_sig, 'geo2.html')



































