단순한 스팸 분류기에서 실용적인 이메일 관리 도구로 업그레이드 - 이메일 스팸분류+메일 요약 시스템

원격 서버 swc

## VS Code + Remote - SSH + Docker 개발 환경 구축 하기

**단계 1: VS Code 확장 프로그램 설치 (팀원 모두)**

1. **VS Code 실행**
2. **확장 탭 열기:** 왼쪽 사이드바에서 네모 박스 4개 모양의 'Extensions' 아이콘을 클릭
3. **필수 확장 설치:** 검색창에 다음 확장 이름을 입력하고 설치
    - `Remote - SSH` (Microsoft): 원격 서버에 SSH로 접속하게 해주는 핵심 확장
    - `Python` (Microsoft): 파이썬 개발에 필수적인 기능(코드 자동 완성, 디버깅 등)을 제공
    - `Docker` (Microsoft): Docker 컨테이너를 VS Code에서 관리하고 컨테이너 내부로 접속하게 해줌
    - `Pylance` (Microsoft): Python 확장의 언어 서버로, 더 나은 코드 분석과 자동 완성 기능을 제공
    - `Prettier - Code formatter` (Esben Fs): 코드 포맷팅을 자동으로 해줘서 코드 스타일을 통일하는 데 도움이 된다. (선택 사항이지만 추천)

**단계 2: 원격 서버 준비 (팀 리더 또는 인프라 담당 팀원)**

- **원격 서버 확보:**
    - 클라우드 서비스(AWS EC2 등)에서 가상 머신 인스턴스를 생성
    
- **SSH 접속 설정:**
    - 서버에 SSH 서버가 설치되어 있는지 확인하고 (기본적으로 설치되어 있음).
    - 팀원들이 각자의 로컬 PC에서 이 서버로 SSH 접속할 수 있도록 `ssh-keygen`으로 키 페어를 생성하고, 서버의 `~/.ssh/authorized_keys` 파일에 각자의 공개 키를 등록.
    - (비밀번호 입력 방식보다 키 방식이 보안상 더 좋고 편리)
    - **팁:** `~/.ssh/config` 파일을 설정하면 SSH 접속이 더 편리해짐.

**단계 3: 원격 서버에 Docker 설치 (팀 리더 또는 인프라 담당 팀원)**

1. **SSH로 서버 접속:** 로컬 터미널에서 `ssh [서버 사용자명]@[서버 IP 주소]` 명령어로 원격 서버에 접속
2. **Docker 설치:** 리눅스 환경에 맞는 Docker 설치 가이드(예: Ubuntu에 Docker 설치)를 따라서 Docker를 설치
3. **Docker 권한 설정:** 일반 사용자도 `sudo` 없이 Docker 명령어를 사용할 수 있도록 사용자 그룹에 Docker 그룹을 추가해 줘.Bash
    
    `sudo usermod -aG docker $USER
    newgrp docker # 변경사항 즉시 적용`
    
    (이후 SSH 세션을 끊고 다시 접속해야 적용될 수 있어.)
    

**단계 4: VS Code에서 Remote - SSH로 원격 서버 접속 (팀원 모두)**

1. **VS Code 왼쪽 하단 아이콘 클릭:** VS Code 왼쪽 하단에 있는 `< >` 모양의 'Remote Indicator' 아이콘을 클릭해 (또는 `F1` 누르고 `Remote-SSH: Connect to Host...` 검색).
2. **SSH 호스트 추가/선택:** 'Add New SSH Host...'를 선택하고 서버 접속 정보(`ssh [사용자명]@[IP 주소]`)를 입력하거나, 이미 설정된 호스트가 있다면 선택해.
3. **접속:** 비밀번호나 SSH 키를 입력하면 원격 서버에 접속돼. VS Code 창이 새로 열리면서 원격 서버 파일 시스템이 보일 거야.

**단계 5: Dockerfile 작성 및 개발 컨테이너 빌드/실행 (팀원 B - MLOps 환경 담당)**

1. **프로젝트 폴더 생성:** 원격 서버에 프로젝트를 위한 폴더를 만들어 (예: `/home/ubuntu/spam_classifier_mlops`).
2. **Dockerfile 작성:** 이 폴더 안에 `Dockerfile`을 만들어. 여기에 파이썬 버전, 필요한 라이브러리(numpy, pandas, scikit-learn, fastapi, uvicorn, transformers 등), 시스템 패키지 등을 정의해.Dockerfile
    
    `# Dockerfile 예시
    FROM python:3.9-slim-buster # 파이썬 버전과 OS 선택 (가볍고 안정적)
    
    WORKDIR /app # 작업 디렉토리 설정# 필요한 시스템 패키지 설치 (예: 텍스트 처리에 필요한 빌드 도구)
    RUN apt-get update && apt-get install -y \
        build-essential \
        # 필요한 다른 패키지들...
        && rm -rf /var/lib/apt/lists/*
    
    # Python 의존성 파일 복사 및 설치
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # 프로젝트 코드 복사
    COPY . .
    
    # API 서버 실행 명령어 (FastAPI 예시)
    # CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    # 개발 환경에서는 CMD를 비워두고 나중에 수동으로 컨테이너 내부에서 실행할 수도 있음`
    
3. **`requirements.txt` 작성:** 프로젝트에 필요한 파이썬 라이브러리 목록을 적어. (예: `scikit-learn`, `fastapi`, `uvicorn`, `transformers`, `mlflow` 등)
4. **Docker 이미지 빌드:** VS Code 터미널(원격 서버에 연결된)에서 `docker build -t spam-classifier-dev-env .` 명령어로 이미지를 빌드해.
5. **Docker 컨테이너 실행:** `docker run -it --name spam_dev_container -v $(pwd):/app spam-classifier-dev-env /bin/bash` 명령어로 컨테이너를 실행해. (여기서 `v $(pwd):/app`은 로컬 프로젝트 폴더를 컨테이너 `/app`에 마운트해서 코드 변경이 실시간 반영되도록 해줘).

**단계 6: VS Code에서 Docker 컨테이너 접속 및 개발 (팀원 모두)**

1. **VS Code 왼쪽 하단 아이콘 클릭:** 다시 `< >` 모양의 'Remote Indicator' 아이콘을 클릭해.
2. **'Attach to Running Container...' 선택:** 실행 중인 `spam_dev_container`를 선택해.
3. **컨테이너 내부 접속:** VS Code 창이 새로 열리면서 이제 Docker 컨테이너 내부의 파일 시스템이 보일 거야. 여기서부터 코드를 작성하고 실행하면 돼. 모든 팀원이 동일한 컨테이너 환경에서 작업하게 되는 거지.

**단계 7: `.vscode` 설정 파일 공유 (팀 리더 또는 환경 담당 팀원)**

1. **`.vscode` 폴더 생성:** 프로젝트 루트 폴더에 `.vscode` 폴더를 만들어.
2. **`settings.json` 작성:**JSON
    
    `// .vscode/settings.json 예시
    {
        "python.defaultInterpreterPath": "/usr/local/bin/python", // 컨테이너 내 파이썬 경로
        "editor.tabSize": 4, // 탭 간격 4칸
        "editor.formatOnSave": true, // 저장 시 자동 포맷팅
        "python.formatting.provider": "black", // Black 포매터 사용
        "python.linting.pylintEnabled": true, // Pylint 린터 사용
        "files.exclude": {
            "**/.git": true,
            "**/.DS_Store": true,
            "**/__pycache__": true,
            "**/.venv": true,
            "**/.pytest_cache": true
        }
    }`
    
3. **`extensions.json` 작성:**JSON
    
    `// .vscode/extensions.json 예시
    {
        "recommendations": [
            "ms-python.python",
            "ms-python.vscode-pylance",
            "ms-azuretools.vscode-docker",
            "esbenp.prettier-vscode",
            "ms-vscode-remote.remote-ssh"
        ]
    }`
    
4. **Git에 커밋:** 이 `.vscode` 폴더를 Git 저장소에 커밋해서 팀원들과 공유해.Bash
    
    `git add .vscode/
    git commit -m "chore: Add VS Code recommended extensions and settings"
    git push`
    
    팀원들이 이 프로젝트를 열면 VS Code가 자동으로 추천 확장 설치를 제안하고, 설정이 적용
    

## 팀 전체 공통 (가장 먼저 해야 할 일)

### 프로젝트 초기화 및 Git 클론

격 서버에 SSH로 접속한 다음, 

GitHub에 만들어 둔 팀 저장소(`spam_classifier_mlops`)를 `git clone`

`cd spam_classifier_mlops`로 해당 폴더로 이동
이 폴더 안에 `.vscode` 폴더를 만들고 

`settings.json`, `extensions.json` 파일을 넣고 Git에 커밋해서 푸시

`requirements.txt` 파일도 프로젝트 루트에 만든다.

### 개발 컨테이너 실행

이 `spam_classifier_mlops` 폴더 안에서 `Dockerfile`을 작성하고 Docker 이미지를 빌드

컨테이너 실행

```python
docker run -it -v $(pwd):/app --name spam_dev_container -p 8000:8000 spam-classifier-dev-env /bin/bash
```

(여기서 `-p 8000:8000`은 API 서버 포트 포워딩)

모든 팀원이 VS Code에서 Remote - SSH로 원격 서버에 접속한 후, 이 컨테이너에 `Attach to Running Container...`해서 개발을 시작

---

**1. 팀원 A: 스팸 메일 분류 모델 개발 (계속 진행)**

- **모델 개발 및 코드 정리:**
    - 현재 만들고 있는 스팸 메일 분류 모델 코드를 `src/models/spam_classifier.py` 같은 파일로 깔끔하게 정리
    - 모델 학습, 예측, 평가하는 함수를 명확하게 분리
    - **목표:** 최소한의 데이터로 스팸/정상 메일을 분류할 수 있는 **작동하는 모델**을 만드는 게 첫 번째 목표
- **데이터셋 준비:**
    - 사용할 스팸 메일 데이터셋(예: CSV 파일)을 `data/raw/` 폴더에 넣어 둠
    - 데이터 전처리 스크립트도 `src/data/preprocess.py` 같은 곳에 만들어 둠
- **Git 커밋:** 작업한 코드를 주기적으로 Git에 커밋하고 푸시 (예: `feat: Implement basic spam classification model`)

---

**2. 팀원 B: 스팸 분류 API 서버 구축 및 CI/CD 초기 설정**

- **FastAPI 기반 API 서버 구축:**
    - `api/main.py` 파일을 만들어서 FastAPI 앱을 초기화해.
    - 팀원 A가 만든 스팸 분류 모델을 로드해서 예측 결과를 반환하는 API 엔드포인트(예: `/classify`)를 구현해.
    - **목표:** `/classify` 엔드포인트에 메일 텍스트를 보내면, '스팸' 또는 '정상'이라는 응답을 받을 수 있도록 만들어.
    - **VS Code 컨테이너 내에서 테스트:** 컨테이너 터미널에서 `uvicorn api.main:app --host 0.0.0.0 --port 8000`으로 서버를 띄우고, `curl`이나 웹 브라우저로 테스트해 봐.
- **Docker 환경 최적화 (팀원들과 공유):**
    - 팀원들이 모두 동일한 개발 컨테이너 환경에서 작업할 수 있도록 `Dockerfile`을 계속 업데이트하고 `requirements.txt`를 관리해.
- **기본 CI/CD 워크플로우 설정 (GitHub Actions):**
    - `.github/workflows/main.yml` 파일을 만들어서, 코드가 `main` 브랜치에 푸시될 때 자동으로 파이썬 코드 린트(lint) 검사나 간단한 테스트가 실행되도록 설정해 봐.
    - **목표:** 코드 푸시 시 GitHub Actions가 자동으로 빌드 과정을 시작하는 것을 확인하는 것.
- **Git 커밋:** 작업한 코드를 주기적으로 Git에 커밋하고 푸시해. (예: `feat: Implement spam classification API with FastAPI`, `chore: Setup basic CI/CD workflow`)

---

**3. 팀원 C: 텍스트 요약 모델 연동 및 MLflow 초기 설정**

- **텍스트 요약 모델 연동 준비:**
    - `src/models/text_summarizer.py` 파일을 만들어서 Hugging Face `transformers` 라이브러리를 이용해 사전 학습된 요약 모델(예: `google/pegasus-cnn_dailymail` 또는 `facebook/bart-large-cnn`)을 로드하고 텍스트를 요약하는 함수를 구현해.
    - **목표:** 파이썬 스크립트에서 스팸 메일 텍스트를 입력하면 요약문이 출력되는 것을 확인하는 것.
- **MLflow 초기 설정 및 실험 추적:**
    - MLflow의 개념(Tracking, Projects, Models, Registry)을 학습해.
    - 팀원 A가 만든 스팸 분류 모델의 학습 과정을 MLflow로 트래킹하도록 코드를 수정해 봐. (예: `mlflow.log_param`, `mlflow.log_metric`, `mlflow.sklearn.log_model` 등)
    - **목표:** `mlflow ui` 명령어로 MLflow UI를 띄워서 스팸 분류 모델의 학습 결과(정확도, F1-score 등)가 기록되는 것을 확인하는 것.
- **Git 커밋:** 작업한 코드를 주기적으로 Git에 커밋하고 푸시해. (예: `feat: Integrate pre-trained text summarization model`, `chore: Setup MLflow tracking for model training`)

---

**4. 팀원 D: 데이터 관리 계획 및 모니터링 초기 설정**

- **데이터셋 상세 분석 및 정리:**
    - 팀원 A가 사용하는 스팸 메일 데이터셋의 특징(컬럼, 데이터 분포, 결측치 등)을 더 깊이 분석해.
    - 데이터 전처리 과정을 이해하고, 필요한 추가 전처리 단계가 있는지 팀원 A와 논의해.
    - `data/processed/` 폴더에 전처리된 데이터가 저장될 계획을 세워.
- **모니터링 기술 학습 및 계획:**
    - Prometheus와 Grafana의 기본 개념(메트릭 수집, 시각화)을 학습해.
    - 어떤 지표(API 요청 수, 응답 시간, 모델 정확도 등)를 모니터링할지 팀원 B, C와 논의하고 목록을 만들어.
    - **목표:** Prometheus와 Grafana를 로컬(또는 원격 서버)에 설치하고, 간단한 시스템 지표(예: 서버 CPU 사용률)를 Grafana 대시보드에서 시각화하는 것을 목표로 해.
- **Git 커밋:** 작업한 내용을 문서화해서 커밋하고 푸시해. (예: `docs: Analyze spam dataset characteristics`, `chore: Research and plan monitoring setup with Prometheus/Grafana`)

---
