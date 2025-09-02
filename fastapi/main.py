from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List
import tritonclient.http as httpclient

import os
import time
import numpy as np
from utils import rain_variables, prediction, result_to_geojson, extract_rain_in_api
from config import *

start_time = time.time()
app = FastAPI()
client = httpclient.InferenceServerClient(url=CLIENT_URL)  # docker-compose 사용 시 서비스명

mask_array = np.load(MASK_PATH)

class SimulationRequest(BaseModel):
    rain: List[float]
    batch_size: int

class RealtimeRequest(BaseModel):
    forecast_hour: int
    batch_size: int

prepare_time = time.time()
print(f"준비 시간 : {prepare_time - start_time}")    

@app.post("/simulation")
def simulation(req: SimulationRequest):
    ready_time = time.time()
    rain = req.rain
    batch_size = req.batch_size
    rain_feat = rain_variables(rain)
    rain_time = time.time()
    print(f"강수 데이터 전처리 시간 : {rain_time - ready_time}")    

    if not os.path.exists(FEATURE_PATH):
        raise HTTPException(status_code=500, detail="features_patched.h5 not found")
    mask = prediction(client, FEATURE_PATH, rain_feat, batch_size)
    mask[mask_array == 1] = 0
    predict_time = time.time()
    print(f"모델 예측 시간 : {predict_time - rain_time}")    

    geojson_str = result_to_geojson(
        mask,
        #REF_HEIGHT, 
        #REF_WIDTH, 
        TRANSFORM, 
        CRS
    )
    posterior_time = time.time()
    print(f"Geojson 변경 시간 : {posterior_time - predict_time}") 
    return Response(content=geojson_str, media_type='application/json')

@app.post("/realtime")
def realtime(req: RealtimeRequest):
    ready_time = time.time()
    forecast_hour = req.forecast_hour
    batch_size = req.batch_size
    
    rain = extract_rain_in_api(forecast_hour)
    rain_feat = rain_variables(rain)
    print(rain_feat)
    rain_time = time.time()
    print(f"강수 데이터 전처리 시간 : {rain_time - ready_time}") 

    if not os.path.exists(FEATURE_PATH):
        raise HTTPException(status_code=500, detail="features_patched.h5 not found")
    mask = prediction(client, FEATURE_PATH, rain_feat, batch_size)
    mask[mask_array == 1] = 0
    predict_time = time.time()
    print(f"모델 예측 시간 : {predict_time - rain_time}")  

    geojson_str = result_to_geojson(
        mask,
        #REF_HEIGHT, 
        #REF_WIDTH, 
        TRANSFORM, 
        CRS
    )
    posterior_time = time.time()
    print(f"Geojson 변경 시간 : {posterior_time - predict_time}") 
    return Response(content=geojson_str, media_type='application/json')