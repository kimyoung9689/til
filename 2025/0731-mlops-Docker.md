## 모니터링 및 로깅

Docker 컨테이너에서 발생하는 로그를 AWS의 **CloudWatch Logs** 서비스로 보내는 방법을 사용
이렇게 하면 API의 동작을 중앙에서 쉽게 확인하고 관리할 수 있다.

**이 작업을 위해 필요한 것:**

1. **EC2 인스턴스에 CloudWatch 접근 권한 부여:** 네 EC2 인스턴스가 CloudWatch Logs로 로그를 보낼 수 있도록 IAM(Identity and Access Management) 역할을 생성하고 인스턴스에 연결
2. **Docker 로깅 드라이버 설정:** Docker가 CloudWatch로 로그를 보내도록 설정
3. **Docker 컨테이너 재실행:** 변경된 로깅 설정을 적용하기 위해 컨테이너를 다시 실행

**단계 1: IAM 역할 생성 및 EC2 인스턴스에 연결**

네 EC2 인스턴스가 CloudWatch Logs에 로그를 보낼 수 있도록 권한을 부여하는 작업

1. **AWS Management Console 로그인:**
    - 웹 브라우저로 AWS 콘솔에 로그인
2. **IAM 대시보드로 이동:**
    - 서비스 검색창에 `IAM`을 입력하거나, `모든 서비스`에서 `IAM`을 찾아 클릭
3. **역할(Roles) 생성:**
    - 왼쪽 메뉴에서 **`역할(Roles)`** 을 클릭
    - **`역할 생성(Create role)`** 버튼을 클릭
4. **신뢰할 수 있는 엔티티 선택:**
    - `AWS 서비스`를 선택하고, `EC2`를 선택
    - **`다음(Next)`** 버튼을 클릭
5. **권한 정책 추가:**
    - 검색창에 `CloudWatch`를 입력
    - `CloudWatchLogsFullAccess` 정책을 찾아 **선택** (이 정책은 CloudWatch Logs에 모든 권한을 부여하므로, 실습용으로 편리하지만 실제 운영 환경에서는 필요한 최소한의 권한만 부여하는 것이 좋디.)
    - **`다음(Next)`** 버튼을 클릭
6. **역할 이름 지정 및 생성:**
    - `역할 이름(Role name)`에 **`EC2-CloudWatch-Logger`**  같이 알아보기 편한거 입력
    - `역할 생성(Create role)` 버튼을 클릭
7. **EC2 인스턴스에 역할 연결:**
    - EC2 대시보드로 이동해. (서비스 검색창에 `EC2` 입력)
    - 왼쪽 메뉴에서 **`인스턴스(Instances)`** 를 클릭
    - 네 스팸 분류 API가 실행 중인 **EC2 인스턴스(`3.35.21.187` 인스턴스)** 를 선택
    - 상단 메뉴에서 **`작업(Actions)`** -> **`보안(Security)`** -> **`IAM 역할 수정(Modify IAM role)`**  클릭
    - `IAM 역할(IAM role)` 드롭다운 메뉴에서 방금 생성한 **`EC2-CloudWatch-Logger`** 역할을 선택
    - **`IAM 역할 업데이트(Update IAM role)`** 버튼을 클릭
