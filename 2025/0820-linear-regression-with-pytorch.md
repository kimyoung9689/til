import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn # 신경망 모델에 쓰이는 도구

# CPU를 사용하도록 설정 ( gpu호환이 안됨)
device = torch.device('cpu')

# CPU에 데이터를 만들고 옮겨줘
x = torch.rand(5, 3).to(device)

# CPU 위에서 간단한 연산하기
y = x * 2

# 결과 확인
print(f"CPU로 연산한 결과:\n{y}")
print(f"결과가 있는 장치: {y.device}")

#학습할 데이터 생성하기

# 학습할 데이터 x
x_train = torch.Tensor([[1],[2],[3]])

# 정답 y
y_train = torch.Tensor([[2],[4],[6]])

plt.scatter(x_train, y_train)



# 선형 회귀 모델
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # 입력차원 1, 출력차원 1

    def forward(self, x):
        return self.linear(x)

# 모델 객체 생성
model = LinearRegressionModel()

print(model)
print(list(model.parameters()))

# 손실함수
criterion = nn.MSELoss()

# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)




# 훈련 루프(Training Loop)
# 훈련 루프는 아래 4단계를 계속 반복

# 예측: 모델이 입력 데이터로 예측값 생성

# 손실 계산: 예측값과 정답이 얼마나 다른지 계산

# 기울기 초기화 및 역전파: 이전에 계산된 기울기를 초기화하고,
# 손실을 줄이기 위해 파라미터를 어떻게 바꿔야 할지 기울기를 계산

# 파라미터 업데이트: 계산된 기울기를 바탕으로
# 모델의 파라미터(W, b)를 업데이트


# 에포크(Epoch) 수 설정: 전체 데이터를 몇 번 반복해서 학습할지 정해.
nb_epochs = 1000

# 훈련 루프 시작
for epoch in range(nb_epochs):
    # 1. 모델 예측값 계산 (Forward Pass)
    prediction = model(x_train)

    # 2. 손실 계산
    loss = criterion(prediction, y_train)

    # 3. 기울기 초기화 (Backpropagation 전)
    # 이전 단계에서 계산된 기울기 값을 0으로 초기화해.
    optimizer.zero_grad()

    # 4. 역전파 (Backpropagation)
    # 손실에 대한 기울기를 계산해.
    loss.backward()

    # 5. 파라미터 업데이트 (Optimizer Step)
    # 계산된 기울기를 바탕으로 가중치와 편향을 업데이트해.
    optimizer.step()

    # 100번마다 손실 값 출력
    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs} Loss: {loss.item():.4f}')

# 학습된 파라미터 값 확인

# 모델의 가중치(W)와 편향(b) 값 출력
print(list(model.parameters()))

# 학습된 모델이 예측한 값 시각화
predicted = model(x_train).detach().numpy()

# 학습 데이터와 모델 예측값을 그래프로 그리기
plt.scatter(x_train, y_train, label='Actual Data')
plt.plot(x_train, predicted, 'r', label='Fitted Line')
plt.legend()
plt.show()


# 학습된 모델 저장하기

# 모델 상태를 딕셔너리 형태로 저장
torch.save(model.state_dict(), 'linear_regression_model.pth')



# 모델을 저장할 때와 동일한 클래스로 객체 생성
loaded_model = LinearRegressionModel()

# 저장된 가중치와 편향 불러오기
loaded_model.load_state_dict(torch.load('linear_regression_model.pth'))

# 5개의 새로운 테스트 데이터 생성
x_test = torch.tensor([[4.0], [5.0], [6.0], [7.0], [8.0]])
y_test = torch.tensor([[8.0], [10.0], [12.0], [14.0], [16.0]])

# 테스트 데이터로 예측하고 손실 계산하기

# 모델을 평가 모드로 전환
model.eval()

with torch.no_grad():
    test_prediction = model(x_test)
    test_loss = criterion(test_prediction, y_test)

print(f'Test Loss: {test_loss.item():.4f}')



이제 이미지 분류(다층 퍼셉트론)와 CNN 고성능 이미지분류를 내일 실시할 예정
