import pandas as pd
import requests
import time
import numpy as np
import os

# 카카오 REST API 키 (새로운 키로 변경)
KAKAO_API_KEY = "38666cb33e533e2bab517d3cff49b371" 
GEOCODING_URL = "https://dapi.kakao.com/v2/local/search/address.json"

# API 호출을 위한 헤더 설정 (정확한 KakaoAK)
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 저장될 파일 경로 설정
project_root_directory = '/root/AIBootcamp14-ml-project1-kimyoung9689-fork/project1'
data_directory = os.path.join(project_root_directory, 'data')
os.makedirs(data_directory, exist_ok=True)
processed_dt_path = os.path.join(data_directory, 'train_xy.csv')

# --- 0. dt 데이터 로드 또는 이어하기 ---
if os.path.exists(processed_dt_path):
    print(f"저장 파일 '{processed_dt_path}'에서 데이터 불러오기...", flush=True)
    dt = pd.read_csv(processed_dt_path)
    print("데이터 불러오기 완료. 이어서 작업합니다.", flush=True)
else:
    print("저장 파일이 없습니다. 현재 메모리의 dt 데이터로 시작합니다. (train.csv 로드 필요 시 아래 주석 해제)", flush=True)
    # dt = pd.read_csv(os.path.join(data_directory, 'train.csv')) 
    # 필요한 초기 컬럼 설정 등 전처리도 여기서 다시 해야 할 수 있음.

# --- 1. XY 결측치 채우기 함수 정의 (개별 성공/실패 메시지 출력 비활성화) ---
def get_coordinates_from_address(address, original_idx=None, attempt_type=""):
    params = {"query": address}
    try:
        response = requests.get(GEOCODING_URL, headers=headers, params=params)
        response.raise_for_status() 
        doc = response.json()

        if doc['documents']:
            x = float(doc['documents'][0]['x'])
            y = float(doc['documents'][0]['y'])
            # print(f"[성공] 인덱스 {original_idx} / 유형 '{attempt_type}' / 주소 '{address}'", flush=True) # 이 부분 주석 처리
            return x, y
        else:
            # print(f"[실패] 인덱스 {original_idx} / 유형 '{attempt_type}' / 주소 '{address}'에 대한 검색 결과 없음.", flush=True) # 이 부분 주석 처리
            return np.nan, np.nan
    except requests.exceptions.RequestException as e:
        # print(f"[오류] 인덱스 {original_idx} / 유형 '{attempt_type}' / 주소 '{address}' API 호출 중 오류 발생: {e}", flush=True) # 이 부분 주석 처리
        return np.nan, np.nan
    except Exception as e:
        # print(f"[오류] 인덱스 {original_idx} / 유형 '{attempt_type}' / 주소 '{address}' 좌표 파싱 중 오류 발생: {e}", flush=True) # 이 부분 주석 처리
        return np.nan, np.nan

# --- 2. 결측치 채우기 전 초기 상태 확인 ---
print("--- XY 결측치 채우기 전 ---", flush=True)
print(f"dt의 '좌표X' 결측치: {dt['좌표X'].isnull().sum()}개", flush=True)
print(f"dt의 '좌표Y' 결측치: {dt['좌표Y'].isnull().sum()}개", flush=True)


# '본번', '부번', '번지' 컬럼을 안전하게 가져오기 위한 헬퍼 함수
def get_safe_num(row, col_name):
    if col_name in row.index and pd.notna(row[col_name]):
        return str(row[col_name]).strip().replace('.0', '')
    return ''

# --- 3. 남은 XY 결측치 채우기 (데이터 전처리 마스터 최종 버전) ---
nan_coords_df = dt[dt['좌표X'].isnull() | dt['좌표Y'].isnull()].copy()

if not nan_coords_df.empty:
    print(f"\n--- API로 채울 XY 결측치 {len(nan_coords_df)}개 처리 시작 (지번/도로명 최적화) ---", flush=True)
    filled_count = 0
    total_to_fill = len(nan_coords_df)

    PRINT_INTERVAL = 1000 
    SAVE_INTERVAL = 2000 

    for i, (idx, row) in enumerate(nan_coords_df.iterrows()):
        x, y = np.nan, np.nan 
        
        sigungu_full = str(row['시군구']).strip() if pd.notna(row['시군구']) else ''
        city_gu = ""
        parts = sigungu_full.split()
        if len(parts) >= 2:
            city_gu = f"{parts[0]} {parts[1]}"
        elif len(parts) == 1:
            city_gu = parts[0]
        dong_name = ""
        if len(parts) >= 3:
            dong_name = parts[2]

        road_name = str(row['도로명']).strip() if pd.notna(row['도로명']) else ''
        apt_name = str(row['아파트명']).strip() if pd.notna(row['아파트명']) else ''
        
        main_num = get_safe_num(row, '본번')
        sub_num = get_safe_num(row, '부번')
        jibun_bunch = get_safe_num(row, '번지')

        # 주소 조합 시도 순서 (가장 가능성 높은 조합부터 시도)
        attempts = []

        # 1. 지번 주소 + 아파트명 (최우선, 삼성래미안 케이스 반영)
        if apt_name and sigungu_full and dong_name and jibun_bunch:
            jibun_address_part_full = f"{jibun_bunch}"
            if sub_num and sub_num != '0':
                jibun_address_part_full += f"-{sub_num}"
            attempts.append((f"{city_gu} {dong_name} {jibun_address_part_full} {apt_name}".strip(), "지번+아파트명(최우선)"))

            # 지번-부번이 너무 길면 부번 없이 시도 (예: 91-5-5 -> 91-5)
            if '-' in jibun_bunch and jibun_bunch.count('-') > 0: 
                base_jibun_bunch = jibun_bunch.split('-')[0]
                if len(jibun_bunch.split('-')) > 1:
                     base_jibun_bunch += f"-{jibun_bunch.split('-')[1]}"
                
                if base_jibun_bunch != jibun_address_part_full: 
                    attempts.append((f"{city_gu} {dong_name} {base_jibun_bunch} {apt_name}".strip(), "지번(간략)+아파트명"))


        # 2. 도로명 주소 + 아파트명
        if apt_name and road_name:
            road_address_part = f"{road_name}"
            if not any(char.isdigit() for char in road_name.split()[-1]): 
                if main_num and main_num != '0':
                    road_address_part += f" {main_num}"
                    if sub_num and sub_num != '0':
                        road_address_part += f"-{sub_num}"
            attempts.append((f"{city_gu} {road_address_part} {apt_name}".strip(), "도로명+아파트명"))
            
            # 건물 번지 없는 도로명 + 아파트명 (예: 언주로 시영)
            attempts.append((f"{city_gu} {road_name} {apt_name}".strip(), "도로명(간략)+아파트명"))


        # 3. 순수 지번 주소 (아파트명 없이)
        if sigungu_full and dong_name and jibun_bunch:
            jibun_address_part_no_apt = f"{jibun_bunch}"
            if sub_num and sub_num != '0':
                jibun_address_part_no_apt += f"-{sub_num}"
            attempts.append((f"{city_gu} {dong_name} {jibun_address_part_no_apt}".strip(), "지번단독"))
            
            # 지번-부번 간략화 시도
            if '-' in jibun_bunch and jibun_bunch.count('-') > 0:
                base_jibun_bunch = jibun_bunch.split('-')[0]
                if len(jibun_bunch.split('-')) > 1:
                     base_jibun_bunch += f"-{jibun_bunch.split('-')[1]}"
                if base_jibun_bunch != jibun_address_part_no_apt:
                    attempts.append((f"{city_gu} {dong_name} {base_jibun_bunch}".strip(), "지번단독(간략)"))

        # 4. 순수 도로명 주소 (아파트명 없이)
        if road_name:
            road_address_part_no_apt = f"{road_name}"
            if main_num and main_num != '0':
                if not any(char.isdigit() for char in road_name.split()[-1]):
                    road_address_part_no_apt += f" {main_num}"
                    if sub_num and sub_num != '0':
                        road_address_part_no_apt += f"-{sub_num}"
            attempts.append((f"{city_gu} {road_address_part_no_apt}".strip(), "도로명단독"))
            
            # 건물 번지 없는 도로명 주소 (예: 언주로)
            attempts.append((f"{city_gu} {road_name}".strip(), "도로명단독(간략)"))

        # 5. 시군구 + 동 + 아파트명만 (최후의 수단)
        if apt_name and sigungu_full and dong_name:
             attempts.append((f"{city_gu} {dong_name} {apt_name}".strip(), "동+아파트명단독"))

        # 6. 아파트명만 (가장 불확실)
        if apt_name and sigungu_full:
            attempts.append((f"{city_gu} {apt_name}".strip(), "아파트명단독"))
            
        # 7. 시군구 + 동 만 (진짜 마지막)
        if sigungu_full and dong_name:
            attempts.append((f"{city_gu} {dong_name}".strip(), "동단독"))

        # 각 주소 조합 시도
        for address_to_query, attempt_type in attempts:
            if not address_to_query or address_to_query.strip() == "":
                continue

            current_x, current_y = get_coordinates_from_address(address_to_query, original_idx=idx, attempt_type=attempt_type)
            
            if not pd.isna(current_x) and not pd.isna(current_y):
                x, y = current_x, current_y
                break 
            
            time.sleep(0.5) # API 호출 제한을 피하기 위한 딜레이

        # 좌표를 찾았으면 데이터프레임 업데이트
        if not pd.isna(x) and not pd.isna(y):
            dt.loc[idx, '좌표X'] = x
            dt.loc[idx, '좌표Y'] = y
            filled_count += 1
        
        # 주기적인 진행 상황 출력
        if (i + 1) % PRINT_INTERVAL == 0: 
             print(f"  {filled_count}/{i+1}개 (전체 {total_to_fill}개 중) 처리 완료. 현재 결측치: {dt['좌표X'].isnull().sum()}개", flush=True)

        # 주기적인 저장
        if (i + 1) % SAVE_INTERVAL == 0:
            dt.to_csv(processed_dt_path, index=False)
            print(f"  [중간 저장] {filled_count}개 처리 후 파일 저장 완료: {processed_dt_path}", flush=True)

    print(f"--- API로 {filled_count}개의 XY 결측치 채우기 완료 ---", flush=True)
    dt.to_csv(processed_dt_path, index=False) 
    print(f"\n최종 처리된 훈련 데이터 저장 완료: {processed_dt_path}", flush=True)

else:
    print("\nAPI로 채울 XY 결측치가 없습니다. 모든 결측치가 채워졌거나 애초에 없었습니다.", flush=True)

# --- 4. 처리 후 최종 상태 확인 ---
print("\n--- XY 결측치 채우기 후 최종 ---", flush=True)
print(f"dt의 '좌표X' 결측치: {dt['좌표X'].isnull().sum()}개", flush=True)
print(f"dt의 '좌표Y' 결측치: {dt['좌표Y'].isnull().sum()}개", flush=True)

# 불필요한 임시 컬럼 삭제
dt.drop(columns=['번지_str', '부번_str', 'jibun_address_part'], inplace=True, errors='ignore') 

print("\ndt 데이터프레임의 '좌표X', '좌표Y' 결측치 처리 작업이 완료됐어.", flush=True)
