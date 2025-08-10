스트레스 지수 예측 프로젝트
1. 대회 개요
대회명: 데이콘 Basic 스트레스 지수 예측
목표: 건강 데이터를 활용하여 개인의 스트레스 지수를 예측하는 모델 개발
평가 지표: MAE (Mean Absolute Error)
참가 규칙: 외부 데이터 사용 불가, AutoML 패키지 사용 불가, 개인 참여.
2. 데이터 분석 및 전처리 (EDA)
2.1 데이터 개요 및 결측치 현황
데이터셋 크기: train과 test 데이터 모두 3,000개의 샘플로 구성.
결측치: medical_history, family_medical_history, edu_level, mean_working 컬럼에서 높은 비율의 결측치를 확인.
결측치 패턴: medical_history와 family_medical_history 결측치가 높은 상관관계를 보였으며, 이는 결측치 자체가 의미있는 정보가 될 수 있음을 시사.
2.2 전처리 및 이상치 처리 전략
이상치 처리: bone_density의 음수 값은 0으로 대체.
결측치 처리:
medical_history, family_medical_history, edu_level: 'unknown' 카테고리로 채움.
mean_working: smoke_status와 edu_level을 기준으로 그룹별 중앙값으로 채움.
인코딩: 모든 범주형 변수에 대해 원-핫 인코딩을 적용.
스케일링: 수치형 변수들의 스케일을 맞추기 위해 StandardScaler를 사용.
3. 특징 공학 (Feature Engineering) 및 모델링
3.1 효과적인 파생 변수
여러 실험을 통해 모델 성능을 향상시킨 핵심 파생 변수는 다음과 같습니다.

체질량지수(BMI): weight / (height / 100) ** 2
맥압(Pulse Pressure): systolic_blood_pressure - diastolic_blood_pressure
콜레스테롤-혈당 비율: cholesterol / (glucose + 1)
혈압 교호 작용: systolic_blood_pressure * diastolic_blood_pressure
혈압 합산: systolic_blood_pressure + diastolic_blood_pressure
3.2 모델 튜닝 및 최적화
모델: LightGBM과 XGBoost 모델을 사용.
하이퍼파라미터 튜닝: RandomizedSearchCV와 교차 검증을 통해 최적의 파라미터를 탐색.
변수 선택: 변수 중요도 분석을 통해 sleep_pattern_sleep difficulty와 같이 중요도가 낮은 변수들을 제거.
4. 최종 모델 및 결과
최종 모델: 하이퍼파라미터 튜닝을 마친 XGBoost 단일 모델을 최종 모델로 선정.
최고 점수: 대시보드 RMSE 점수 0.15293 달성.
노트북 관리: 점수 개선이 있을 때마다 새로운 노트북 파일(예: notebooks/submission_01.ipynb, notebooks/submission_02.ipynb)을 만들어 기록하는 방식으로 진행.
5. 기술 스택
언어: Python
주요 라이브러리: Pandas, NumPy, Scikit-learn, XGBoost, LightGBM
환경: Jupyter Notebook
