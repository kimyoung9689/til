퍼블릭주소는 다시 시작했을때 접근 주소가 새로 바뀐다.

edit session 클릭해서 퍼블릭주소 변경 리모트호스트 안에 넣어주면 됨 OK

docker ps 하면 어제 내용 안보이는게 정상

그 이유는 인스턴트 중단때문(시스템 다운) 도커 컨테이너는 프로세스기 때문에 종료됨

docker ps -a 해주면 다시 뜸 이걸 다시 실행시켜주면 된다.

docker start my-mlops

docker exec -it my-mlops bash 다시 도커로 진입


그 외 모델 추론 과정은 노션 페이지 참고(너무 길다)
