# Seaborn 기초부터 끝까지 총정리

## 1. Seaborn 개요

### 특징
- matplotlib 기반의 통계 데이터 시각화 라이브러리
- 아름답고 정보가 풍부한 통계 그래프 제공
- pandas DataFrame과 완벽 호환
- 통계적 관계를 쉽게 시각화
- 내장된 테마와 색상 팔레트 제공

### 설치 및 임포트
```python
pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
```

## 2. 기본 설정

### 스타일 설정
```python
sns.set_style("whitegrid")    # whitegrid, darkgrid, white, dark, ticks
sns.set_palette("husl")       # 색상 팔레트 설정
sns.set_context("notebook")   # paper, notebook, talk, poster
```

### 색상 팔레트
```python
sns.color_palette()           # 기본 팔레트
sns.color_palette("viridis")  # 연속형 팔레트
sns.color_palette("Set1")     # 범주형 팔레트
sns.hls_palette(8, l=.3, s=.8) # HLS 팔레트
```

## 3. 관계형 플롯 (Relational Plots)

### scatterplot (산점도)
```python
sns.scatterplot(data=df, x="total_bill", y="tip")
sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex", size="size")
```

### lineplot (선 그래프)
```python
sns.lineplot(data=df, x="timepoint", y="signal")
sns.lineplot(data=df, x="timepoint", y="signal", hue="event", style="event")
```

### relplot (관계형 플롯의 통합)
```python
sns.relplot(data=df, x="total_bill", y="tip", kind="scatter")
sns.relplot(data=df, x="total_bill", y="tip", col="time", hue="smoker")
```

## 4. 범주형 플롯 (Categorical Plots)

### stripplot (스트립 플롯)
```python
sns.stripplot(data=df, x="day", y="total_bill")
sns.stripplot(data=df, x="day", y="total_bill", jitter=True, hue="sex")
```

### swarmplot (스웜 플롯)
```python
sns.swarmplot(data=df, x="day", y="total_bill")
sns.swarmplot(data=df, x="day", y="total_bill", hue="sex")
```

### boxplot (박스 플롯)
```python
sns.boxplot(data=df, x="day", y="total_bill")
sns.boxplot(data=df, x="day", y="total_bill", hue="smoker")
```

### violinplot (바이올린 플롯)
```python
sns.violinplot(data=df, x="day", y="total_bill")
sns.violinplot(data=df, x="day", y="total_bill", hue="sex", split=True)
```

### barplot (막대 그래프)
```python
sns.barplot(data=df, x="sex", y="total_bill")
sns.barplot(data=df, x="sex", y="total_bill", hue="smoker")
```

### pointplot (포인트 플롯)
```python
sns.pointplot(data=df, x="sex", y="survived", hue="class")
```

### countplot (카운트 플롯)
```python
sns.countplot(data=df, x="class")
sns.countplot(data=df, x="class", hue="who")
```

### catplot (범주형 플롯의 통합)
```python
sns.catplot(data=df, x="day", y="total_bill", kind="box")
sns.catplot(data=df, x="day", y="total_bill", col="time", kind="violin")
```

## 5. 분포 플롯 (Distribution Plots)

### histplot (히스토그램)
```python
sns.histplot(data=df, x="total_bill")
sns.histplot(data=df, x="total_bill", hue="sex", multiple="stack")
sns.histplot(data=df, x="total_bill", kde=True)
```

### kdeplot (커널 밀도 추정)
```python
sns.kdeplot(data=df, x="total_bill")
sns.kdeplot(data=df, x="total_bill", hue="sex")
sns.kdeplot(data=df, x="total_bill", y="tip")  # 2D KDE
```

### ecdfplot (경험적 누적분포함수)
```python
sns.ecdfplot(data=df, x="total_bill")
sns.ecdfplot(data=df, x="total_bill", hue="sex")
```

### rugplot (러그 플롯)
```python
sns.rugplot(data=df, x="total_bill")
```

### displot (분포 플롯의 통합)
```python
sns.displot(data=df, x="total_bill", kind="hist")
sns.displot(data=df, x="total_bill", col="time", kde=True)
```

## 6. 회귀 플롯 (Regression Plots)

### regplot (회귀 플롯)
```python
sns.regplot(data=df, x="total_bill", y="tip")
sns.regplot(data=df, x="total_bill", y="tip", order=2)  # 2차 회귀
```

### lmplot (선형 모델 플롯)
```python
sns.lmplot(data=df, x="total_bill", y="tip")
sns.lmplot(data=df, x="total_bill", y="tip", hue="smoker")
sns.lmplot(data=df, x="total_bill", y="tip", col="time")
```

### residplot (잔차 플롯)
```python
sns.residplot(data=df, x="total_bill", y="tip")
```

## 7. 행렬 플롯 (Matrix Plots)

### heatmap (히트맵)
```python
sns.heatmap(df.corr())
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
sns.heatmap(df.corr(), mask=np.triu(np.ones_like(df.corr())))
```

### clustermap (클러스터 맵)
```python
sns.clustermap(df.corr())
sns.clustermap(df.corr(), annot=True, cmap="viridis")
```

## 8. 다중 플롯

### FacetGrid (면 분할 그리드)
```python
g = sns.FacetGrid(df, col="time", row="smoker", margin_titles=True)
g.map(sns.histplot, "total_bill")
```

### PairGrid (쌍 그리드)
```python
g = sns.PairGrid(df, vars=["total_bill", "tip", "size"])
g.map_diag(sns.histplot)
g.map_upper(sns.scatterplot)
g.map_lower(sns.kdeplot)
```

### pairplot (쌍 플롯)
```python
sns.pairplot(df)
sns.pairplot(df, hue="species")
sns.pairplot(df, diag_kind="kde")
```

## 9. 통계 추정 및 오차

### 신뢰구간 표시
```python
sns.lineplot(data=df, x="timepoint", y="signal", ci=95)
sns.barplot(data=df, x="sex", y="total_bill", ci="sd")
```

### 부트스트래핑
```python
sns.barplot(data=df, x="sex", y="total_bill", n_boot=1000)
```

## 10. 색상과 스타일 커스터마이징

### 색상 팔레트 생성
```python
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
sns.set_palette(colors)
```

### 커스텀 컬러맵
```python
from matplotlib.colors import LinearSegmentedColormap
colors = ["red", "white", "blue"]
cmap = LinearSegmentedColormap.from_list("custom", colors)
sns.heatmap(data, cmap=cmap)
```

### 스타일 매개변수
```python
sns.set_style("whitegrid", {"grid.linewidth": 0.5})
sns.despine(left=True, bottom=True)  # 축 제거
```

## 11. 축과 레이블 설정

### 축 범위 설정
```python
plt.xlim(0, 50)
plt.ylim(0, 10)
```

### 레이블과 제목
```python
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Tips vs Total Bill")
```

### 범례 설정
```python
plt.legend(title="Gender", loc="upper right")
```

## 12. 서브플롯과 피겐 크기

### 피겐 크기 설정
```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="total_bill", y="tip")
```

### 서브플롯 생성
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(data=df, x="total_bill", ax=axes[0])
sns.boxplot(data=df, x="day", y="total_bill", ax=axes[1])
```

## 13. 애니메이션과 인터랙티브

### 애니메이션 (matplotlib과 연동)
```python
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()

def animate(frame):
    ax.clear()
    subset = df[df['frame'] <= frame]
    sns.scatterplot(data=subset, x="x", y="y", ax=ax)

ani = FuncAnimation(fig, animate, frames=100, interval=50)
```

## 14. 데이터 전처리 팁

### 결측값 처리
```python
df_clean = df.dropna()
df_filled = df.fillna(df.mean())
```

### 데이터 변환
```python
df['log_total_bill'] = np.log(df['total_bill'])
df['tip_rate'] = df['tip'] / df['total_bill']
```

### 범주형 데이터 순서 지정
```python
day_order = ['Thur', 'Fri', 'Sat', 'Sun']
sns.boxplot(data=df, x="day", y="total_bill", order=day_order)
```

## 15. 고급 기능

### 조건부 색상 매핑
```python
palette = {"Male": "skyblue", "Female": "lightcoral"}
sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex", palette=palette)
```

### 마커 스타일
```python
sns.scatterplot(data=df, x="total_bill", y="tip", style="time", markers=["o", "s"])
```

### 투명도 설정
```python
sns.scatterplot(data=df, x="total_bill", y="tip", alpha=0.6)
```

### 선 스타일
```python
sns.lineplot(data=df, x="timepoint", y="signal", linestyle="--", linewidth=2)
```

## 16. 성능 최적화

### 큰 데이터셋 처리
```python
# 샘플링
df_sample = df.sample(n=1000)
sns.scatterplot(data=df_sample, x="x", y="y")

# 빈닝
sns.histplot(data=df, x="value", bins=50)
```

### 메모리 효율성
```python
# 불필요한 컬럼 제거
df_viz = df[['x', 'y', 'category']].copy()
```

## 17. 저장과 내보내기

### 그래프 저장
```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.savefig('plot.pdf', format='pdf')
plt.savefig('plot.svg', format='svg')
```

### 고해상도 저장
```python
plt.figure(figsize=(12, 8), dpi=300)
sns.scatterplot(data=df, x="x", y="y")
plt.savefig('high_res_plot.png', dpi=300, bbox_inches='tight')
```

## 18. 내장 데이터셋

### 자주 사용되는 데이터셋
```python
tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
mpg = sns.load_dataset("mpg")
```

## 19. 실전 활용 예제

### 탐색적 데이터 분석
```python
# 기본 정보 확인
sns.pairplot(df, hue="target")

# 상관관계 분석
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0)

# 분포 확인
sns.displot(df, x="value", col="category", kind="kde")
```

### 시계열 데이터 시각화
```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x="date", y="value", hue="category")
plt.xticks(rotation=45)
```

### 범주형 데이터 분석
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
sns.countplot(data=df, x="category", ax=axes[0,0])
sns.boxplot(data=df, x="category", y="value", ax=axes[0,1])
sns.violinplot(data=df, x="category", y="value", ax=axes[1,0])
sns.barplot(data=df, x="category", y="value", ax=axes[1,1])
```

## 20. 문제 해결 및 디버깅

### 자주 발생하는 오류
- 데이터 타입 불일치: astype() 사용
- 결측값: dropna() 또는 fillna() 사용
- 메모리 부족: 데이터 샘플링 또는 청크 처리

### 성능 개선
- 불필요한 데이터 제거
- 적절한 플롯 타입 선택
- 색상 팔레트 최적화

### 호환성 문제
- matplotlib 버전 확인
- pandas 버전 호환성
- 환경별 폰트 설정
