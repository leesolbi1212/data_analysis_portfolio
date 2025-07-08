# interactivegraph.py
# 인터렉티브 그래프

import pandas as pd
import plotly.express as px

# 데이터
mpg = pd.read_csv('assets/mpg.csv')
#mpg.info()

# 산점도 그래프
# fig = px.scatter(data_frame=mpg, x='cty', y='hwy', color='drv')
# fig.write_html('scatter.html', auto_open=True)

# 막대 그래프
# df = mpg.groupby('category', as_index=False) \
#         .agg(n=('category', 'count'))
# fig2 = px.bar(data_frame=df, x='category', y='n', color='category')
# fig2.write_html('bar.html', auto_open=True)

# 선 그래프
# economics = pd.read_csv('assets/economics.csv')
# fig3 = px.line(data_frame=economics, x='date', y='psavert')
# fig3.write_html('line.html', auto_open=True)

# 상자 그래프
# fig4 = px.box(data_frame=mpg, x='drv', y='hwy', color='drv')
# fig4.write_html('box.html', auto_open=True)

























