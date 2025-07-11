from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import tritonclient.http as httpclient

import os
import rasterio
from utils import rain_variables, prediction, result_to_geojson, extract_rain_in_api
from config import *

app = FastAPI()
client = httpclient.InferenceServerClient(url=CLIENT_URL)  # docker-compose 사용 시 서비스명


with rasterio.open(REF_PATH) as src:
    ref_height = src.height
    ref_width = src.width
    transform = src.transform
    crs = src.crs
    

class SimulationRequest(BaseModel):
    rain: List[float]
    batch_size: int

class RealtimeRequest(BaseModel):
    forecast_hour: int
    batch_size: int


@app.post("/simulation")
def simulation(req: SimulationRequest):
    rain = req.rain
    batch_size = req.batch_size
    rain_feat = rain_variables(rain)

    if not os.path.exists(FEATURE_PATH):
        raise HTTPException(status_code=500, detail="features_patched.h5 not found")
    mask = prediction(FEATURE_PATH, rain_feat, batch_size)

    geojson_str = result_to_geojson(mask)
    return Response(content=geojson_str, media_type='application/json')

@app.post("/realtime")
def realtime(req: RealtimeRequest):
    time = req.forecast_hour
    batch_size = req.batch_size
    
    rain = extract_rain_in_api(time)
    rain_feat = rain_variables(rain)

    if not os.path.exists(FEATURE_PATH):
        raise HTTPException(status_code=500, detail="features_patched.h5 not found")
    mask = prediction(FEATURE_PATH, rain_feat, batch_size)

    geojson_str = result_to_geojson(mask)
    return Response(content=geojson_str, media_type='application/json')