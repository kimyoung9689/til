목표: 새로운 데이터가 들어오거나 코드 변경이 있을 때 모델 학습 과정을 자동으로 >재실행하는 파이프라인을 구축하는 단계.
진행 상황: ci-cd.yml 파일을 통해 GitHub Actions 워크플로우를 구축해서 notebooks/train_model.py가 특정 파일 변경 시 자동으로 실행되도록 함. GitHub Actions에서 실패한 이력은 있지만, 파이프라인 자체는 구축됐다고 볼 수 있다.
모델 서빙/배포 (Model Serving & Deployment)
목표: 학습된 모델을 실제 서비스에서 사용할 수 있도록 API 형태로 배포하는 단계.
진행 상황:
FastAPI를 사용하여 스팸 분류 API 서버(api/main.py)를 개발했고, 학습된 모델을 이 API에 성공적으로 통합.
Docker를 사용하여 이 API 서버를 컨테이너화하는 api/Dockerfile도 만들었음.
가장 최근에 로컬 환경에서 Docker 이미지를 빌드하고 컨테이너를 실행해서 API가 스>팸/햄 예측을 올바르게 하는 것까지 확인 (이 부분이 방금 완료된 중요한 단계)
