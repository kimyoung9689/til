NumPy 문법
1. 배열 생성 (Array Creation)
기본 배열 생성

import numpy as np

# 1차원 배열
arr_1d = np.array([1, 2, 3, 4, 5])

# 2차원 배열
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

# 3차원 배열
arr_3d = np.array([[[1, 2], [3, 4]], 
                   [[5, 6], [7, 8]]])
배열 생성 함수들
python
# 영 배열 (zeros)
zeros_arr = np.zeros((3, 4))        # 3x4 영 배열
zeros_1d = np.zeros(5)              # 1차원 영 배열

# 일 배열 (ones)
ones_arr = np.ones((2, 3))          # 2x3 일 배열
ones_1d = np.ones(4)                # 1차원 일 배열

# 단위 행렬 (identity matrix)
eye_arr = np.eye(3)                 # 3x3 단위행렬
identity_arr = np.identity(4)       # 4x4 단위행렬

# 범위 배열
range_arr = np.arange(0, 10, 2)     # 0부터 10미만까지 2씩 증가
linspace_arr = np.linspace(0, 1, 5) # 0부터 1까지 5개 균등분할

# 랜덤 배열
random_arr = np.random.random((2, 3))    # 0~1 사이 랜덤값
random_int = np.random.randint(1, 10, 5) # 1~10 사이 정수 5개

# 빈 배열 (메모리만 할당)
empty_arr = np.empty((2, 2))

# 특정 값으로 채운 배열
full_arr = np.full((3, 3), 7)       # 3x3 배열을 7로 채움
2. 배열 속성 (Array Properties)
python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 배열 모양 (shape)
print(arr.shape)        # (2, 3) - 2행 3열

# 차원 수 (ndim)
print(arr.ndim)         # 2 - 2차원

# 데이터 타입 (dtype)
print(arr.dtype)        # int64 (시스템에 따라 다름)

# 총 원소 개수 (size)
print(arr.size)         # 6

# 메모리 크기 (itemsize, nbytes)
print(arr.itemsize)     # 8 (바이트)
print(arr.nbytes)       # 48 (바이트)
3. 인덱싱 & 슬라이싱 (Indexing & Slicing)
1차원 배열
python
arr_1d = np.array([1, 2, 3, 4, 5])

# 인덱싱
print(arr_1d[0])        # 1 (첫 번째 원소)
print(arr_1d[-1])       # 5 (마지막 원소)

# 슬라이싱
print(arr_1d[1:4])      # [2 3 4] (1번부터 3번까지)
print(arr_1d[::2])      # [1 3 5] (0번부터 끝까지 2씩 건너뛰기)
print(arr_1d[::-1])     # [5 4 3 2 1] (역순)
2차원 배열
python
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 인덱싱
print(arr_2d[0, 2])     # 3 (0행 2열)
print(arr_2d[1, 0])     # 4 (1행 0열)

# 슬라이싱
print(arr_2d[0, :])     # [1 2 3] (0행 전체)
print(arr_2d[:, 2])     # [3 6] (2열 전체)
print(arr_2d[0:, 1:])   # [[2 3] [5 6]] (0행부터, 1열부터)
불린(Boolean) 인덱싱
python
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 3
print(arr[mask])        # [4 5] (3보다 큰 값들)
print(arr[arr % 2 == 0]) # [2 4] (짝수들)
4. 기본 연산 (Basic Operations)
산술 연산
python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 배열 간 연산
add_result = arr1 + arr2    # [5 7 9]
sub_result = arr1 - arr2    # [-3 -3 -3]
mul_result = arr1 * arr2    # [4 10 18] (원소별 곱셈)
div_result = arr2 / arr1    # [4. 2.5 2.]

# 스칼라와 연산
scalar_mul = arr1 * 2       # [2 4 6]
scalar_add = arr1 + 10      # [11 12 13]
power_result = arr1 ** 2    # [1 4 9]
수학 함수들
python
arr = np.array([1, 4, 9, 16])

# 제곱근
sqrt_result = np.sqrt(arr)  # [1. 2. 3. 4.]

# 지수/로그
exp_result = np.exp(arr)    # 지수함수
log_result = np.log(arr)    # 자연로그

# 삼각함수
sin_result = np.sin(arr)
cos_result = np.cos(arr)
5. 브로드캐스팅 (Broadcasting) ⭐
브로드캐스팅은 다른 크기의 배열 간에도 연산을 가능하게 해주는 중요한 개념입니다!

브로드캐스팅 규칙
뒤쪽 차원부터 비교
차원의 크기가 같거나, 둘 중 하나가 1이면 브로드캐스가 가능
크기가 1인 차원은 다른 차원에 맞춰 확장됨
python
# 예시 1: 1차원 + 스칼라
arr1 = np.array([1, 2, 3])
result1 = arr1 + 5          # [6 7 8]

# 예시 2: 2차원 + 1차원
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
arr1d = np.array([10, 20, 30])
result2 = arr2d + arr1d     # [[11 22 33]
                            #  [14 25 36]]

# 예시 3: 서로 다른 모
arr_col = np.array([[1],    # (3, 1) 모양
                    [2],
                    [3]])
arr_row = np.array([4, 5, 6])  # (3,) 모양

result3 = arr_col + arr_row    # [[5 6 7]
                               #  [6 7 8]
                               #  [7 8 9]]
브로드캐스팅 시각화
arr_col (3,1)     arr_row (3,)
[[1]       +      [4, 5, 6]
 [2]
 [3]]

브로드캐스팅 후:
[[1, 1, 1]   +   [[4, 5, 6]   =   [[5, 6, 7]
 [2, 2, 2]        [4, 5, 6]        [6, 7, 8]
 [3, 3, 3]]       [4, 5, 6]]       [7, 8, 9]]
6. 유용한 함수들
집계 함수
python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 전체 배열에 대한 집계
print(np.sum(arr))      # 21 (전체 합)
print(np.mean(arr))     # 3.5 (평균)
print(np.max(arr))      # 6 (최대값)
print(np.min(arr))      # 1 (최소값)

# 축(axis)별 집계
print(np.sum(arr, axis=0))   # [5 7 9] (열별 합)
print(np.sum(arr, axis=1))   # [6 15] (행별 합)
배열 변형
python
arr = np.array([1, 2, 3, 4, 5, 6])

# 모양 변경
reshaped = arr.reshape(2, 3)    # 2x3으로 변형
flattened = reshaped.flatten()  # 1차원으로 평탄화

# 전치 (transpose)
transposed = reshaped.T         # 행과 열 바꾸기
 핵심 포인트
배열 생성: np.array(), np.zeros(), np.ones(), np.eye()
속성 확인: .shape, .ndim, .dtype, .size
인덱싱: arr[행, 열] 형태로 접근
브로드캐스팅: 다른 크기 배열도 연산 가능! (가장 중요)
연산: +, -, *, /, ** 모두 원소별 연산
 실전 팁
Alt + Shift + 방향키: 줄 전체 복사
Alt + 방향키: 줄 이동
Shift + Home: 커서 앞부분 선택
Shift + End: 커서 뒷부분 선택
다음 시간: 넘파이와 판다스 심화 학습


