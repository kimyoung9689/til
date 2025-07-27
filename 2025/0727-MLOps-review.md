MLOps 데이터 수집 및 전처리 핵심 정리
📋 프로젝트 개요
목표: TMDB API를 활용해 영화 데이터를 수집하고, 추천 시스템을 위한 사용자 시청 로그 데이터를 생성
🏗️ 시스템 아키텍처
환경 구성

컨테이너: Docker (Python 3.11-bookworm)
작업 디렉토리: /opt/data-prepare
결과 저장: /opt/data-prepare/result

핵심 모듈 구조
/opt/data-prepare/
├── crawler.py          # TMDB API 데이터 수집
├── preprocessing.py    # 데이터 전처리 및 가상 로그 생성
├── main.py            # 메인 실행 진입점
├── .env               # 환경변수 (API 키 등)
└── result/            # 결과 데이터 저장
    ├── popular.json   # 원천 영화 데이터
    └── watch_log.csv  # 최종 시청 로그 데이터
 핵심 컴포넌트
1. TMDBCrawler 클래스
python# 주요 기능
- get_popular_movies(): 페이지별 인기 영화 수집
- get_bulk_popular_movies(): 다중 페이지 일괄 수집
- save_movies_to_json_file(): JSON 파일로 저장
2. TMDBPreProcessor 클래스
python# 핵심 로직
- augmentation(): 평점 기반 가중치 적용 (2^rating)
- selection(): 사용자별 랜덤 콘텐츠 선택 (최대 20개)
- generate_watch_second(): 평점 기반 시청 시간 생성
 데이터 파이프라인
Input (TMDB API)

소스: movie-popular-list API (1페이지)
파라미터: 한국 지역(KR), 한국어(ko-KR)

Processing (피처 엔지니어링)

가상 사용자 생성: 100명
콘텐츠 선택: 평점 높을수록 선택 확률 증가
시청 시간: 평점과 연동된 지수 함수 기반 생성 (0~7200초)

Output (최종 데이터)
csvuser_id,content_id,watch_seconds,rating,popularity
 핵심 알고리즘
가중치 기반 선택
python# 평점이 높을수록 더 많이 선택되도록 augmentation
count = int(pow(2, rating))  # 2^평점
시청 시간 생성
python# 평점과 연동된 지수 함수 + 노이즈
base_time = max_runtime * (base^(rating-5) - base^-5) / (base^5 - base^-5)
watch_second = base_time + noise
 보안 및 환경 관리

환경변수 분리: .env 파일로 API 키 관리
템플릿 제공: .env.template으로 설정 가이드
요청 제한: 0.4초 간격으로 API 호출 제한

 MLOps 관점에서의 의미

재현 가능성: 시드 고정 (random.seed(0))
확장성: 페이지 범위, 사용자 수 등 파라미터화
모듈화: 크롤링-전처리-실행 단계 분리
환경 독립성: Docker 컨테이너 기반 실행

