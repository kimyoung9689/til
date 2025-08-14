# **2. 데이터 세트 및 데이터 로더**

데이터를 깔끔하게 관리하는 방법

PyTorch는 `Dataset`과 `DataLoader`라는 두 가지 도구를 사용해서 데이터를 쉽게 다룬다.

**`Dataset`**       : 딥러닝에 사용할 **데이터와 정답을 저장**
**`DataLoader`** : `Dataset`에 있는 데이터를 학습에 쓰기 편하게 작은 묶음으로 나눠 모델에게 전달

이 두 가지를 사용하면 데이터 준비 코드와 모델 학습 코드를 분리할 수 있음

PyTorch에서 자주 쓰이는 대표적인 데이터셋

| 분야 | 데이터셋 이름 | 특징 |
| --- | --- | --- |
| **이미지(`torchvision`)** | `MNIST` | 0~9 손글씨 숫자 이미지 (흑백) |
|  | `FashionMNIST` | 옷, 신발 등 패션 아이템 이미지 (흑백) |
|  | `CIFAR10` | 비행기, 고양이 등 10가지 클래스의 컬러 이미지 |
|  | `ImageNet` | 수백만 개의 이미지와 1,000개의 클래스를 가진 초대형 데이터셋 |
| **텍스트(`torchtext`)** | `IMDb` | 영화 리뷰를 긍정/부정으로 분류하는 감성 분석 데이터셋 |
|  | `WikiText` | 위키피디아에서 가져온 대규모 텍스트 데이터셋 |
|  | `AG_NEWS` | 뉴스 기사를 주제별로 분류하는 데이터셋 |
| **오디오(`torchaudio`)** | `SpeechCommands` | "yes", "no" 등 간단한 음성 명령이 담긴 오디오 데이터셋 |
|  | `YesNo` | "yes"와 "no" 두 가지 음성 단어가 담긴 데이터셋 |

딥러닝 학습 과정을 연습하고, 모델의 성능을 평가하는 **연습용**

### 데이터셋 불러오기

```python
import torch  # 모든 기능 불러오기
from torch.utils.data import Dataset # torch안에 Dataset이라는 특정 기능만 가져오기
from torchvision import datasets # PyTorch가 제공하는 이미지 데이터셋 불러올 준비
from torchvision.transforms import ToTensor # 이미지 -> 텐서로 바꿔주는 도구
import matplotlib.pyplot as plt

# 훈련용 데이터셋 불러오는 함수
training_data = datasets.FashionMNIST(
    root="data", # 데이터 파일 data라는 폴더에 저장하라고 지정
    train=True,  # 전체 데이터에서 훈련용 데이터만 가져오기(60,000개)
    download=True, # data 폴더에 데이터가 없으면 자동으로 다운로드
    transform=ToTensor() # 이미지 가져올 때, 텐서로 바꿔서 가져오기
)

# 테스트용 데이터셋 불러오는 함수
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # 테스트용 데이터만 가져오기
    download=True,
    transform=ToTensor()
)
```

### **데이터 세트 반복 및 시각화**

**FashionMNIST** 데이터셋에서 몇 개의 샘플을 뽑아 그림으로 보여주기

```python
# 이미지 레이블(0~9)을 실제 옷 이름으로 매핑하는 딕셔너리
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
# 그림 그릴 도화지 생성
figure = plt.figure(figsize=(8, 8))

# 도화지를 3행 3열, 총9칸으로 나눔
cols, rows = 3, 3

# 1부터 9까지 9번 반복하며 이미지 9개를 그림
for i in range(1, cols * rows + 1):
    # training_data에서 무작위로 한 개의 이미지 인덱스를 선택
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    
    # 선택된 인덱스의 이미지와 레이블을 가져옴
    img, label = training_data[sample_idx]
    
    # 현재 반복 순서에 맞춰 격자에 서브플롯을 추가
    figure.add_subplot(rows, cols, i)
    
    # labels_map을 사용해 숫자 레이블 대신 옷 이름을 제목으로 표시
    plt.title(labels_map[label])
    
    # 그림 주변의 축(눈금)을 숨김
    plt.axis("off")
    
    # 이미지를 흑백으로 변환하여 서브플롯에 표시
    # img.squeeze()는 이미지 차원을 조정해 plt.imshow()에 맞는 형태로 만들어줌
    plt.imshow(img.squeeze(), cmap="gray")

# 완성된 9개의 이미지 출력    
plt.show()
```

### **파일에 대한 사용자 정의 데이터 세트 만들기**

```python
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### **`__init__`**

객체를 초기화하는 역할

클래스를 처음 만들 때 설정을 해줌

```python
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### **`__len__`**

객체에 들어있는 데이터의 **총 개수**를 알려주는 역할

데이터 세트의 샘플 수를 반환

```python
def __len__(self):
    return len(self.img_labels)
```

### **`__getitem__`**

`데이터셋[인덱스]`와 같은 방식으로 데이터를 꺼낼 때 자동으로 호출
`training_data[0]`라고 하면 첫 번째 데이터를 가져오기 위해 이 함수가 작동한다.

```python
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = read_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

### **DataLoaders로 학습을 위한 데이터 준비**

`DataLoader` 역할

`Dataset`에서 데이터를 가져와서 학습에 더 편하게 만들어주는 도구

**미니배치 생성**: 한 번에 여러 개의 데이터(여기서는 64개)를 묶어서 전달
**데이터 섞기**: 매 번 에포크 마다 데이터를 무작위로 섞어서 모델이 데이터의 순서를 외우는 걸 막음
**병렬 처리**: 여러 개의 코어를 사용해서 데이터를 더 빠르게 불러올 수 있게 해줌

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

이 코드는 `training_data`와 `test_data`라는 큰 데이터 덩어리를 

`DataLoader`를 사용해서 64개씩 묶어 미니배치로 나눠주는 역할을 함
모델은 이 꾸러미들을 하나씩 받아서 학습

### **DataLoader를 반복**

`DataLoader`필요에 따라 데이터 세트를 반복할 수 있다.

반복하는 이유 

모델이 데이터를 부담 없이 처리하게 하면서, 데이터 편향 없이 골고루 학습해 성능을 높이기 위함

1. 미니배치 학습 

모델을 학습할 때 데이터 전체를 한 번에 넣는 대신, 데이터를 작은 묶음으로 나눠서 학습

메모리 절약 : 미니배치로 나누면 메모리를 효율적으로 사

빠른 학습  : 데이터를 여러 묶음으로 나눠서 GPU 같은 가속 장치로 병렬 처리해서 학습속도 빠

1. 무작위성 부여

`shuffle=True` 옵션을 주면 매번 데이터를 가져올 때마다 순서를 섞어준다.
과적합 방지 : 만약 데이터 순서가 항상 똑같으면, 모델이 데이터의 순서 패턴을 외워버
일반화 성능 향상 : 순서를 섞어 다양한 데이터를 골고루 학습. 모델이 데이터의 본질적인 특징을 더 잘 배움

```python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

---


