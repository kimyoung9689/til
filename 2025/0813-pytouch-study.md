# PyTorch (파이토치)

# 파이토치란?

파이토치는 페이스북(현 Meta)에서 개발한 오픈소스 머신러닝 라이브러리

파이썬 기반이라 문법이 직관적이고, 연구 및 실험에 특히 많이 사용

# **1. 텐서(Tensor)**

파이토치에서 데이터를 담는 가장 기본적인 단위

넘파이(NumPy)의 `ndarray`와 비슷, GPU 연산을 지원해 딥러닝 모델 학습이 빠름

```python
import torch

# 1차원 텐서 (벡터)
tensor_1d = torch.tensor([1, 2, 3]) 
print(f"1차원 텐서: {tensor_1d}")

# 2차원 텐서 (행렬)
tensor_2d = torch.tensor([[1, 2], [3, 4]])
print(f"2차원 텐서: \n{tensor_2d}")

# 3차원 텐서
tensor_3d = torch.zeros(2, 3, 4) # 2x3x4 크기의 0으로 채워진 텐서
print(f"3차원 텐서 크기: {tensor_3d.shape}")
```

### 넘파이 배열, 텐서 서로 바꾸기

```python
import numpy as np
import torch

# 넘파이 배열 만들기
numpy_array = np.array([[1, 2], [3, 4]])
print(f"넘파이 배열:\n{numpy_array}")

# 넘파이 배열을 파이토치 텐서로 바꾸기
pytorch_tensor = torch.from_numpy(numpy_array)
print(f"\n넘파이에서 만든 텐서:\n{pytorch_tensor}")

# 파이토치 텐서를 넘파이 배열로 바꾸기
numpy_from_pytorch = pytorch_tensor.numpy()
print(f"\n텐서에서 만든 넘파이 배열:\n{numpy_from_pytorch}")
```

### 텐서 속성 복사

```python
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor: \n {x_rand} \n")
```

결과

```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.9776, 0.6797],
        [0.0617, 0.7338]])
```

### `shape`은 **텐서의 차원**을 정의하는 튜플

```python
shape = (2,3,) # 차원을 정의
rand_tensor = torch.rand(shape) # 0과 1사이의 무작위 값으로 채워진 텐서 생성
ones_tensor = torch.ones(shape) # 모든 값이 1인 텐서 생성
zeros_tensor = torch.zeros(shape) # 모든 값이 0인 텐서 생성

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

결과

```python
Random Tensor:
 tensor([[0.9764, 0.2805, 0.5193],
        [0.8189, 0.3358, 0.5709]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

`shape`을 이용하면 원하는 크기의 텐서를 쉽고 빠르게 만들 수 있다.

### 텐서의 주요 속성 알아보기

```python
tensor = torch.rand(3,4)

tensor.shape # 텐서의 모양
tensor.dtype # 텐서에 들어있는 데이터의 타입
tensor.device # 텐서가 지금 CPU와 GPU 중 어디에 저장되어 있는지 알려줌
# cpu는 CPU에, cuda:0은 GPU에 있다는 뜻. 딥러닝에서는 GPU를 많이 쓰니까 이 속성도 중요
```

결과

```python
torch.Size([3, 4])
torch.float32
cpu
```

### 텐서 연산

파이토치에서 텐서를 만들면 기본적으로 CPU 메모리에 저장.

하지만 딥러닝 모델을 학습시킬 때는 GPU의 엄청난 병렬 연산 능력이 필요하니 

텐서를 CPU에서 GPU로 옮겨야함

텐서를 CPU에서 GPU로 옮기려면 `.to()` 메서드를 사용

accelerator를 이용해 사용 가능한 가속기 있으면 사용하는 코드

```python
import torch

# 사용할 장치를 설정할 변수를 만들어.
device = "cpu"

# 'torch.accelerator'를 사용해서 가속기가 있는지 확인.
# 이 코드는 CUDA, MPS, ROCm 등 모든 가속기를 알아서 체크해 줘.
if torch.accelerator.is_available():
    # 가속기가 있다면, 현재 사용 가능한 가속기를 device 변수에 저장해.
    device = torch.accelerator.current_accelerator()
    print(f"가속기를 사용합니다: {device}")
else:
    # 가속기가 없으면 CPU를 사용해.
    print("가속기가 없어 CPU를 사용합니다.")

# 텐서를 만들고, 위에서 설정한 장치(device)로 이동시켜.
tensor = torch.randn(3, 3)
tensor = tensor.to(device)

print(f"\n최종 텐서의 장치: {tensor.device}")
```

하나하나 있는지 체크하고 있으면 사용하는 코드

```python
import torch

# 사용할 가속기를 저장할 변수
device = None

# CUDA (NVIDIA GPU)가 있는지 확인
if torch.cuda.is_available():
    device = "cuda"
    print("CUDA 가속기를 사용할 수 있습니다.")

# MPS (Apple Silicon)가 있는지 확인
elif torch.backends.mps.is_available():
    device = "mps"
    print("MPS 가속기를 사용할 수 있습니다.")

# ROCm (AMD GPU)가 있는지 확인
elif hasattr(torch.backends, 'rocm') and torch.backends.rocm.is_available():
    device = "rocm"
    print("ROCm 가속기를 사용할 수 있습니다.")
    
# Intel 가속기 (XPU)가 있는지 확인 (파이토치 공식 라이브러리가 아닌 별도 패키지 필요)
# elif hasattr(torch, 'xpu') and torch.xpu.is_available():
#     device = "xpu"
#     print("XPU 가속기를 사용할 수 있습니다.")

# 위에 해당하는 가속기가 없으면 CPU 사용
else:
    device = "cpu"
    print("가속기를 사용할 수 없으므로 CPU를 사용합니다.")

# 텐서 생성 및 가속기로 이동
# 예시 텐서 생성
tensor = torch.randn(3, 3)

# 위에서 확인한 device 변수에 따라 텐서를 해당 가속기로 이동
tensor = tensor.to(device)

print(f"\n텐서가 위치한 장치: {tensor.device}")
```

대표적인 가속기 종류

| 가속기 종류 | 주로 사용되는 하드웨어 | 어떤 상황에 쓰는지 | 장점 | 단점 |
| --- | --- | --- | --- | --- |
| **CUDA** | **NVIDIA GPU** | 딥러닝 모델 학습 및 추론. **가장 일반적**이고 폭넓게 사용됨. | 성능이 뛰어나고, 관련 라이브러리 및 커뮤니티가 가장 활발함. | NVIDIA GPU에서만 사용 가능. |
| **MPS** | **Apple 실리콘 (M1, M2 칩 등)** | 맥(Mac) 컴퓨터에서 딥러닝 모델 학습 및 추론. | 애플 기기에 최적화되어 있어, 메모리 공유 등 효율성이 좋음. | 아직 CUDA만큼 기능이 다양하지 않고, 지원하는 모델이 제한적일 수 있음. |
| **ROCm** | **AMD GPU** | AMD GPU를 사용하는 환경에서 딥러닝. | AMD 하드웨어에서 높은 성능을 낼 수 있음. | CUDA만큼 커뮤니티나 라이브러리 지원이 활발하지 않음. |
| **XPU** | **Intel GPU** | 인텔 GPU를 사용하는 환경에서 딥러닝. | 인텔 하드웨어에서 잘 작동함. | 아직 발전 단계라 CUDA나 MPS만큼 널리 쓰이지는 않음. |

### Numpy와 유사한 인덱싱 및 슬라이싱

```python
tensor = torch.ones(4, 4)                # 모든 값이 1로 채워진 4x4 크기의 텐서
print(f"First row: {tensor[0]}")         # 첫 번째 행 출력
print(f"First column: {tensor[:, 0]}")   # 모든 행의 첫 번째 열 출력
print(f"Last column: {tensor[..., -1]}") # 마지막 열 출력
tensor[:,1] = 0                          # 두 번째 열의 모든 값을 0으로 변경
print(tensor)  # 변경된 텐서를 출력하여 두 번째 열이 0으로 바뀐 것을 보여줌
```

PyTorch에서 텐서의 특정 데이터를 쉽게 가져오거나 수정하는 방법

결과

```python
First row: tensor([1., 1., 1., 1.])
First column: tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

### 텐서 결합 (`torch.cat`)

`torch.cat()` 함수는 여러 텐서를 주어진 차원(`dim`)을 따라 합치는 데 사용

텐서들을 단순히 이어 붙일 때 사용하는 기능

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1) # tensor 3번 반복 리스트 묶고 열 방향 결합
print(t1)
```

결과

```python
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

`torch.stack` 함수는 `torch.cat`와 비슷하지만, 새로운 차원을 만들어 텐서를 쌓는 방식이 차이점이다.

`torch.cat`이 실제 사용되는 예시
이미지 처리 (U-Net)
자연어 처리 (NLP)

멀티모달(Multi-modal) 데이터

`torch.cat`은 여러 정보를 한데 모으는 **"풀(glue)"** 같은 역할을 하기 때문에, 

복잡한 모델을 만들수록 활용 빈도가 높다.

이렇게 쓰려면 행과 열의 크기를 맞춰줘야 하는데…!

### 텐서의 모양(shape)을 바꿔주는 기능

- **1.`torch.reshape()` 또는 `torch.view()`**: 텐서의 전체 요소 개수가 유지되는 한,

        원하는 모양으로 크기를 바꿔준다. 예) `4x4` 텐서를 `2x8`로 바꾸기

- **2.`torch.transpose()`**: 텐서의 두 차원(dimension)을 서로 바꿔주는 기능.

        `4x5` 텐서를 `torch.transpose(tensor, 0, 1)`로 바꾸면 `5x4` 텐서가 됨. 

        이게 가장 흔하게 쓰이는 방법

- **3.`torch.unsqueeze()`**: 크기가 1인 새로운 차원을 추가.

        예)  `(5,)` 모양의 텐서를 `(1, 5)`나 `(5, 1)`로 바꿀 수 있다.

```python
import torch

t1 = torch.randn(4, 3) # 4x3 텐서
t2 = torch.randn(3, 4) # 3x4 텐서

# t2를 transpose해서 4x3 텐서로 만듭니다.
t2_transposed = t2.transpose(0, 1)

# 이제 두 텐서의 모양이 4x3으로 같아졌으므로 합칠 수 있습니다.
result = torch.cat([t1, t2_transposed], dim=1)

print(f"원래 t1의 모양: {t1.shape}")
print(f"원래 t2의 모양: {t2.shape}")
print(f"변환된 t2의 모양: {t2_transposed.shape}")
print(f"합쳐진 결과의 모양: {result.shape}")
```

### 행렬 곱셈

```python
# 행렬 곱셈
y1 = tensor @ tensor.T       # @연산자는 행렬 곱셈을 의미
y2 = tensor.matmul(tensor.T) # matmul() 함수는 @와 같은 역할

y3 = torch.rand_like(y1)               # y1과 동일한 크기 텐서 생성
torch.matmul(tensor, tensor.T, out=y3) # out 파라미터 사용해 연산결과 y3텐서에 저장

# 요소별 곱셈
# 텐서의 같은 위치에 있는 요소들끼리 곱하는 연산
# 행렬 곱셈과 다르게 행렬의 크기가 같아야 함
z1 = tensor * tensor     # *연산자는 곱셈을 위미
z2 = tensor.mul(tensor)  # mul() 함수는 *와 같은 역할

z3 = torch.rand_like(tensor)      # tensor와 동일한 크기 텐서 생성
torch.mul(tensor, tensor, out=z3) # out 파라미터 사용해 연산결과 z3텐서에 저장
```

### 단일 요소 텐서

`item()` 함수는 텐서에 요소가 하나만 있을 때, 그 값을 파이썬의 숫자(예: 정수, 실수)로 변환하는 데 사용

```python
agg = tensor.sum()    # 텐서의 모든 요소 더해 하나의 값을 가진 단일 요소 텐서 생성
agg_item = agg.item() # agg 텐서의 값을 파이썬의 숫자 값으로 변환해 agg_item에 저장
print(agg_item, type(agg_item)) # 변환된 숫자 값과 그 데이터 타입을 출력
# 이 코드를 실행하면 텐서가 아닌 일반 파이썬 숫자(예: float)가 출력됨
```

결과

```python
12.0 <class 'float'>
```

`item()`은 딥러닝 모델의 손실값을 계산할 때처럼, 최종적으로 하나의 숫자 결과만 필요할 때 유용하게 사용

### 여러 개의 요소가 있는 텐서를 바꿔줄 땐

- **`.tolist()`**: 텐서를 중첩된 **파이썬 리스트**로 변환
- **`.numpy()`** :  텐서를 **넘파이 배열**로 변환

```python
import torch

tensor = torch.tensor([[1, 2], [3, 4]])

# 리스트로 변환
python_list = tensor.tolist()
print(f"변환된 리스트: {python_list}")
print(f"데이터 타입: {type(python_list)}")

# 넘파이 배열로 변환
numpy_array = tensor.numpy()
print(f"변환된 넘파이 배열:\n{numpy_array}")
print(f"데이터 타입: {type(numpy_array)}")
```

**중요**: 만약 GPU에서 작업 중인 텐서를 변환하고 싶다면, 먼저 `텐서.cpu()`를 사용해서 **CPU로 옮긴 다음** `.tolist()`나 `.numpy()`를 사용해야 함. GPU에 있는 텐서는 바로 변환할 수 없기 때문(`item()`도 같음)

### 제자리 연산

연산 결과를 새로운 변수에 저장하는 대신, 원래 텐서에 바로 저장하는 연산을 말함
이름 뒤에 밑줄(`_`)이 붙는 특징이 있음  예), `x.copy_()`, `x.t_()`

제자리 연산은 메모리를 절약할 수 있다는 장점이 있지만, **미분 값을 계산할 때 문제가 될 수 있다.**

연산 기록이 즉시 사라져버리기 때문에 미분을 추적하기 어려워지기 때문 = 잘 안쓰는게 좋다.

```python
print(f"{tensor} \n")
tensor.add_(5) # 텐서의 모든 요소에 5를 더함
print(tensor)
```

### 넘파이 배열과 텐서의 연동

```python
t = torch.ones(5) # 모든 값이 1인 5개짜리 텐서 생성
print(f"t: {t}") 
n = t.numpy()     # 텐서 t를 넘파이 배열 n으로 바꿔줌
print(f"n: {n}")
```

이렇게 변환된 텐서와 넘파이 배열은 같은 메모리를 공유함

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

결과

```python
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

한쪽의 값을 바꾸면 다른 한쪽의 값도 같이 바뀌는 특징이 있다.
