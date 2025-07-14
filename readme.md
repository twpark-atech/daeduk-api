# 대덕특구 침수 예측 모델 API
## Project
### Workplace Structure
```
daeduk_api
├─fastapi
└─triton
```
### 구축 환경
|                | version |
| ------------------------- | --------------- |
| Platform | AMD64 |
| Docker | 28.3.0 |
| Python | 3.10.18 |
| Ubuntu | 22.04 |
| CPU | i9-12900K |
| RAM | 64GB |
| GPU | RTX 3090 |

### 실행 방법
1. API Key 발급 후 fastapi에서 .env 작성
```
KMA_SFCTM3={기상청API허브 API Key}
GETULTRASRTFCST={기상청 단기예보 조회서비스 API Key}
```
기상청API허브 : https://apihub.kma.go.kr/
기상청 단기예보 조회서비스 : https://www.data.go.kr/data/15084084/openapi.do

2. 데이터 다운로드
```
data -> /fastapi/data
1 -> /triton/flood_model/1
```
데이터 다운로드 : https://drive.google.com/file/d/1auJhLPEoTDKF6LsVlmhiX52bQ0CVrNQP/view?usp=sharing

3. Docker 실행
```
docker compose up -d
```
 
4. URL 접속 후 실행
- '/simulation' 및 '/realtime' 선택
- '/simulation' 선택 시 Parameter
    - 'rain' : 최대 34시간의 시간당 강수강도를 리스트 형태로 입력
        ex) [2.0, 1.0, 7.0, 4.5, 4.0, 0.0, 6.5, 16.5, 53.0, 7.0, 0.0, 0.5, 5.0, 0.0, 0.5, 0.5, 0.5] 
    - 'batch_size' : 사용할 배치 사이즈 입력
        ex) 8 (RTX 3090 기준, Batch Size 8: 12초 가량 소요)
- '/realtime' 선택 시 Parameter
    - 'forecast_hour' : 앞으로 예측할 시간 입력 (0 ~ 6시간)
        ex) 6 (기상청에서 제공해주는 초단기예보는 6시간까지 존재)
    - 'batch_size' : 사용할 배치 사이즈 입력
        ex) 8 (RTX 3090 기준, Batch Size 8: 12초 가량 소요)
URL : localhost:9000/docs
