# scikit-learn (사이킷런)

파이썬으로 머신러닝을 할 때 가장 많이 쓰는 라이브러리

사이킷런이라는 이름은 **"SciPy Toolkit for machine learning"** 의 줄임말

사이파이 기반의 머신러닝 도구 키트 라는 뜻

사이언스 키트(Science Kit)의 줄임말 사이킷

사이킷런, 사이파이, 그리고 넘파이는 서로 긴밀하게 연결된 관계

넘파이 과학 계산 라이브러리

다차원 배열 객체 제공 , 다양한 수학 함수를 지원 , 대규모 수치 연산

사이파이 

넘파이의 배열 구조를 기반으로 만들어진 라이브러리

선형 대수, 최적화, 통계, 푸리에 변환, 신호 처리 등 더 복잡하고 전문적인 과학 계산을 위한 고급 기능제공

넘파이가 기본적인 고성능 배열 연산을 담당하는 기반이라면, 

사이파이는 그 위에 더 복잡한 과학 계산 기능을 추가한 확장판

사이킷런은 이 넘파이와 사이파이를 활용해서 머신러닝 기능을 구현한 라이브러리

`scikit` 시리즈는 `SciPy` 기반의 도구 키트들을 의미
`scikit-learn`과 `scikit-image` 같이 많이 사용

### scikit-image

이미지 처리에 특화된 라이브러리.  `NumPy` 배열을 이미지 데이터로 사용

**주요 기능:**

- **필터링 (Filtering)**: 이미지를 부드럽게,선명하게하고 가장자리를 검출하는 등 다양한 필터를 적용
- **분할 (Segmentation)**: 이미지를 여러 영역으로 나누는 기능. 예) 이미지에서 특정 물체 분리
- **특징 추출 (Feature detection)**: 이미지에서 코너, 질감(texture) 같은 중요한 특징을 찾아냄
- **기하학적 변환 (Geometric transformations)**: 이미지를 회전, 크기 조절, 왜곡시키는 등의 작업.
- **색상 공간 변환 (Color space manipulation)**: 이미지를 흑백으로 바꾸거나, 다른 색상 형식으로 변환

`scikit-image`는 `scikit-learn`처럼 `NumPy`와 `SciPy` 위에 구축되었고, 

비슷한 파이썬적인 문법을 사용하기 때문에 함께 사용하기 좋다.
예) `scikit-image`로 이미지에서 특징 추출, 그 특징을 `scikit-learn` 모델의 입력으로 사용

최근에는 딥러닝 기반의 이미지 처리 라이브러리(예: TensorFlow, PyTorch)가 많이 쓰이고 있지만, 간단하고 효율적인 이미지 처리나 전처리 작업에는 여전히 `scikit-image`가 널리 사용됨

### 주요 특징

- **다양한 머신러닝 알고리즘 제공:** 분류, 회귀, 클러스터링 등 다양한 알고리즘을 쓸 수 있다.
- **간편한 사용법:** 모든 알고리즘이 비슷한 구조로 되어 있어서 쓰기 편함.

         `fit()`, `predict()`, `transform()` 같은 함수들을 주로 사용

- **데이터 전처리 기능:** 결측치 처리, 데이터 스케일링 같은 전처리 기능도 함께 제공
- **풍부한 문서:** 공식 문서가 잘 정리되어 있고 예제도 많아서 배우기 쉬움

주요 모듈

![image.png](attachment:27c012d6-96e2-490c-932e-fa603341be44:image.png)

사이킷런 모듈

- `sklearn.linear_model`: 선형 모델.
- `sklearn.tree`: 결정 트리.
- `sklearn.ensemble`: 앙상블 학습.
- `sklearn.svm`: 서포트 벡터 머신.
- `sklearn.neighbors`: K-최근접 이웃.
- `sklearn.cluster`: 클러스터링.
- `sklearn.model_selection`: 모델 선택 및 평가.
- `sklearn.preprocessing`: 데이터 전처리.
- `sklearn.metrics`: 모델 성능 지표.
- `sklearn.datasets`: 예제 데이터셋.

이 외에도 차원 축소, 신경망 모델, 교차 검증 등 다양한 모듈이 있다.

**기본 사용 흐름**

1. **데이터 준비:** `load_iris` 같은 함수로 데이터를 불러온다.
2. **데이터 분할:** `train_test_split`으로 학습용 데이터와 테스트용 데이터로 나눔
3. **모델 선택 및 학습:** 원하는 모델을 선택하고 `fit()` 함수로 학습
4. **예측 및 평가:** `predict()` 함수로 예측하고 `accuracy_score` 같은 지표로 모델 성능을 평가

### 문법

대부분의 사이킷런 모듈은 비슷한 문법을 사용.  초보자에게  편리함
보통 `from sklearn.모듈이름 import 클래스이름` 형식으로 사용한다.

예를 들어, 선형 회귀 모델을 사용하려면 `linear_model` 모듈에서 `LinearRegression` 클래스를 불러옴

```python
from sklearn.linear_model import LinearRegression

# 모델 객체 생성
model = LinearRegression()

# 데이터로 학습
model.fit(X_train, y_train)

# 예측
predictions = model.predict(X_test)
```

이런 식으로 대부분의 모델은 `fit()`, `predict()` 함수를 사용해서 학습과 예측을 진행

### 사이킷런 연습하기

사이킷런에 익숙해지려면 아래 4가지 단계를 반복해서 연습하기

### 1. 데이터 불러오기

제공된 데이터를 불러와서 `X` (문제지)와 `y` (정답)로 나누는 연습. 

`sklearn.datasets` 모듈에 있는 `load_iris`나 `load_boston` 같은 예제 데이터를 써보면 좋다.

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
```

### 2. 데이터 분할

학습용 데이터와 테스트용 데이터를 나누는 게 중요. 

`model_selection` 모듈의 `train_test_split` 함수를 사용해서 데이터를 7:3, 8:2 비율로 나누는 연습

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. 모델 학습 및 예측

가장 핵심적인 단계. 간단한 모델부터 시작해서 익숙해지기

- **모델 불러오기**: `from sklearn.모듈 import 모델이름`
- **모델 객체 생성**: `model = 모델이름()`
- **모델 학습**: `model.fit(X_train, y_train)`
- **예측**: `y_pred = model.predict(X_test)`

이 과정을 여러 모델(`LinearRegression`, `RandomForestClassifier`, `KMeans` 등)로 반복해봐.

### 4. 모델 평가

모델이 얼마나 잘 예측했는지 확인하기

`metrics` 모듈의 다양한 함수들을 써서 모델 성능을 평가하기

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```

데이터 분할 , 모델학습 중요
