import torch
import torchvision.datasets as dsets # 이미지 데이터 다루는 도구
import torchvision.transforms as transforms # 데이터 모델에 맞게 변환하는 도구
from torch.utils.data import DataLoader # 데이터를 효율적으로 불러오는 도구
import matplotlib.pyplot as plt

# 데이터셋 다운로드 후 전처리
mnist_train = dsets.MNIST(root='MNIST_data/', # MNIST 데이터 다운로드 경로
                          train=True, # 훈련 데이터셋
                          transform=transforms.ToTensor(), # 이미지를 텐서로 변환
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/', # MNIST 데이터 다운로드 경로
                         train=False, # 테스트 데이터셋
                         transform=transforms.ToTensor(), # 이미지를 텐서로 변환
                         download=True)

# 데이터로더 설정
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=100, # 한 번에 100개씩 가져옴
                         shuffle=True, # 데이터 섞기
                         drop_last=True) # 마지막 남은 데이터 버리기


# 이미지 확인
# 첫 번째 이미지와 레이블을 확인
image, label = mnist_train[0]
plt.imshow(image.squeeze().numpy(), cmap='gray')
plt.title(f'Label: {label}')
plt.show()


import torch.nn as nn

# 하이퍼 파라미터 설정
# 모델 성능에 영향을 주지만 학습 과정에서는 변하지 않는 값
training_epochs = 15
learning_rate = 0.001

# 모델 설계
class MNIST_MLP(nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        # 입력층 -> 은닉층1
        self.fc1 = nn.Linear(784, 256, bias=True)
        # 은닉층1 -> 은닉층2
        self.fc2 = nn.Linear(256, 256, bias=True)
        # 은닉층2 -> 출력층
        self.fc3 = nn.Linear(256, 10, bias=True)

    def forward(self, x):
        # 텐서를 펼치는 작업 (flatten)
        x = x.view(-1, 28 * 28)
        # 은닉층에 활성화 함수 ReLU 적용
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        # 출력층
        x = self.fc3(x)
        return x

# 모델 객체 생성
model = MNIST_MLP()


# 모델 평가하기
# 모델을 평가 모드로 전환 (학습 과정 비활성화)
model.eval()

# 기울기 계산 비활성화
with torch.no_grad():
    X_test = mnist_test.test_data.view(-1, 28 * 28).float()
    Y_test = mnist_test.test_labels

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy:', accuracy.item())
