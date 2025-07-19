pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'NanumGothic' # 'NanumGothic'으로 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
# (유니코드 문자가 이뻐보인다고 기본 값으로 쓰는데 한글폰트 대부분이 유니코드 마이너스 기호가 없어서 문제생김)


# Figure (피겨): 가장 큰 도화지
# Axes (액시즈): x축 y축 제목 같은 걸 넣는 그림 영역
# Plot (플랏): 이건 Axes 안에 그려지는 실제 그림


# 1. 그림 그릴 데이터 준비 (x값, y값)
# x는 가로축, y는 세로축에 놓일 숫자들
x_data = [1, 2, 3, 4, 5]
y_data = [2, 4, 6, 8, 10] # x값의 2배씩 늘어나는 숫자

# 2. 그림 그리기 (Figure와 Axes가 자동으로 만들어져)
plt.plot(x_data, y_data) # x_data와 y_data를 가지고 선으로 'plot' 해줘!

# 3. 그림에 이름 붙이기 (제목, 축 이름)
plt.title('나의 첫 번째 선 그래프') # 그림 제목
plt.xlabel('시간 (분)')           # 가로축 이름
plt.ylabel('거리 (미터)')         # 세로축 이름

# 4. 그린 그림을 화면에 출력
plt.show()



# 1. Figure(도화지) 하나랑, 그 위에 Axes(그림 영역) 두 개를 나란히 만들어줘!
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# --- 첫 번째 그림 영역 (ax1)에 그림 그리기: 산점도 (Scatter Plot) ---
student_heights = [160, 165, 170, 175, 180]
student_weights = [55, 60, 65, 70, 75]

ax1.scatter(student_heights, student_weights, color='red', marker='o')
ax1.set_title('키와 몸무게 관계 (산점도)')
ax1.set_xlabel('키 (cm)')
ax1.set_ylabel('몸무게 (kg)')

# --- 두 번째 그림 영역 (ax2)에 그림 그리기: 막대 그래프 (Bar Plot) ---
favorite_colors = ['빨강', '파랑', '초록', '노랑']
num_people = [10, 15, 8, 12]

ax2.bar(favorite_colors, num_people, color=['red', 'blue', 'green', 'yellow'])
ax2.set_title('좋아하는 색깔별 친구 수')
ax2.set_xlabel('색깔')
ax2.set_ylabel('친구 수')

# 그림들이 서로 겹치지 않게 간격 자동 조절
plt.tight_layout()

# 최종 그림을 화면에 보여줘!
plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Figure와 Axes 4개 생성 및 데이터 준비
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
student_scores = np.random.randint(50, 100, 20)
heights = np.random.normal(170, 5, 100)

# 1. 산점도 (왼쪽 위)
axes[0, 0].scatter(range(len(student_scores)), student_scores, color='purple', marker='o')
axes[0, 0].set_title('학생 점수 분포 (산점도)')
axes[0, 0].set_xlabel('학생 번호')
axes[0, 0].set_ylabel('점수')

# 2. 막대 그래프 (오른쪽 위)
months = ['1월', '2월', '3월', '4월']
sales = [100, 150, 120, 180]
axes[0, 1].bar(months, sales, color='skyblue')
axes[0, 1].set_title('월별 판매량')
axes[0, 1].set_xlabel('월')
axes[0, 1].set_ylabel('판매량')

# 3. 히스토그램 (왼쪽 아래)
axes[1, 0].hist(heights, bins=10, color='lightgreen', edgecolor='black')
axes[1, 0].set_title('사람들 키 분포 (히스토그램)')
axes[1, 0].set_xlabel('키 (cm)')
axes[1, 0].set_ylabel('인원수')

# 4. 선 그래프 (오른쪽 아래)
x_line = np.linspace(0, 5, 50)
y_line = x_line ** 2
axes[1, 1].plot(x_line, y_line, color='orange', linestyle='--', linewidth=2, label='y = x²')
axes[1, 1].set_title('제곱 함수 그래프')
axes[1, 1].set_xlabel('X 값')
axes[1, 1].set_ylabel('Y 값')
axes[1, 1].legend()

# 간격 자동 조절 및 그림 표시
plt.tight_layout()
plt.show()



# 1. 그림 그릴 데이터 준비
x = [1, 2, 3, 4, 5]
y = [1, 3, 2, 5, 4]

# 2. 선 그래프 그리기
plt.plot(x, y)

# 3. 그림에 제목과 축 이름 붙이기
plt.title('저장할 예쁜 그림')
plt.xlabel('X 값')
plt.ylabel('Y 값')

# 4. 그린 그림을 'my_beautiful_plot.png'라는 이름의 PNG 파일로 저장해줘!
# dpi=300: 'dots per inch'의 줄임말이야. 그림의 해상도(선명도)를 나타내.
# 숫자가 높을수록 더 선명하게 저장돼. 보통 300이나 600을 많이 써.
plt.savefig('my_beautiful_plot.png', dpi=300)

# 5. 저장한 그림을 화면에도 보여줘! (저장했다고 화면에 안 뜨는 건 아니야)
plt.show()



# 그림 꾸미기 & 강조하기 (디테일 추가!)
# 1. 범례 (Legend) 위치 변경하기

# ax.legend()를 쓰면 label로 지정한 이름들이 그림에 나타나잖아? 이 범례가 기본적으로는 그림의 왼쪽 위에 나타나는데, 그림 내용에 따라 다른 위치로 옮기고 싶을 때가 있을 거야.

# loc (location): 범례의 위치를 지정하는 옵션이야.

# 'upper left' (왼쪽 위, 기본값)

# 'upper right' (오른쪽 위)

# 'lower left' (왼쪽 아래)

# 'lower right' (오른쪽 아래)

# 'center' (가운데)

# 'best' (Matplotlib이 가장 적절한 빈 공간을 찾아줘




import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='사인 함수')
plt.plot(x, y2, label='코사인 함수')
plt.title('사인과 코사인 함수')
plt.xlabel('X 값')
plt.ylabel('Y 값')

# 범례를 오른쪽 아래에 표시해줘!
plt.legend(loc='best')

plt.show()



# 사인: 원 위를 도는 점의 위아래 움직임 (Y축 값)
# 코사인: 원 위를 도는 점의 좌우 움직임 (X축 값)


import matplotlib.pyplot as plt
import numpy as np

# 데이터 준비
x = np.array([1, 2, 3, 4, 5])
y1 = x * 1
y2 = x * 2
y3 = x * 3

plt.figure(figsize=(8, 6)) # 새 그림을 그리고 크기를 정해줘

# 첫 번째 선: 빨간색 두꺼운 점선, 동그라미 마커
plt.plot(x, y1, color='red', linestyle='--', linewidth=2, marker='o', markersize=8, label='데이터 1배')

# 두 번째 선: 파란색 실선, 삼각형 마커, 마커 테두리 초록색, 안쪽 노란색
plt.plot(x, y2, color='blue', linestyle='-', linewidth=1.5, marker='^', markersize=10,
         markeredgecolor='green', markerfacecolor='yellow', label='데이터 2배')

# 세 번째 선: 보라색 점선-실선, 별 마커
plt.plot(x, y3, color='purple', linestyle='-.', linewidth=2.5, marker='*', markersize=12, label='데이터 3배')

plt.title('다양한 선 스타일과 마커')
plt.xlabel('X 값')
plt.ylabel('Y 값')
plt.grid(True) # 그리드 표시
plt.legend() # 범례 표시

plt.show()



import matplotlib.pyplot as plt
import numpy as np # (이 코드에서는 직접 사용 안 하지만, 보통 함께 불러와)

# --- 1. 데이터 준비: 어떤 그림을 그릴지 숫자들을 모아두는 곳 ---
months = ['1월', '2월', '3월', '4월', '5월', '6월'] # 월(가로축)
monthly_sales = [120, 150, 130, 180, 200, 170] # 월별 판매량(세로축)

# --- 2. Figure(도화지)와 Axes(그림 영역) 만들기 ---
# 하나의 그림만 그릴 거니까, plt.figure()로 도화지를 만들고
# plt.plot()으로 바로 그림을 그릴 거야.
plt.figure(figsize=(8, 6)) # 가로 10인치, 세로 6인치 크기의 도화지를 만들어줘

# --- 3. 실제 그림 그리기: 선 그래프 (Line Plot) ---
plt.plot(months, monthly_sales, # x축은 months, y축은 monthly_sales 데이터로 선을 그려줘
         color='royalblue',     # 선의 색깔을 '로열블루'로 해줘
         marker='o',            # 각 데이터 점에 동그라미 모양의 마커를 표시해줘
         linestyle='-',         # 선 스타일을 '실선'으로 해줘 (기본값이기도 해)
         linewidth=2,           # 선의 두께를 2로 해줘
         markersize=8)          # 마커(점)의 크기를 8로 해줘

# --- 4. 그림 꾸미기: 제목, 축 이름, 그리드 등 ---
plt.title('상반기 월별 판매량 추이') # 그림 전체의 제목
plt.xlabel('월')                      # 가로축(x축)의 이름
plt.ylabel('판매량 (만 원)')         # 세로축(y축)의 이름
plt.grid(True,                       # 격자(그리드) 선을 표시해줘
         linestyle='--',             # 그리드 선을 점선으로 해줘
         alpha=0.7)                  # 그리드 선을 살짝 투명하게(70%) 해줘

# --- 5. 그림을 화면에 보여줘! ---
plt.show()



