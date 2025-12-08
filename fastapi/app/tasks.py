# tasks.py

from __future__ import annotations

import os
import time
import numpy as np
from celery import shared_task

from .celery_app import celery_app
from .utils import rain_variables, prediction, result_to_geojson, save_flood_mask_png, extract_flooded_links, extract_rain_in_api
from .config import *
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

TRITON_MODEL_NAME = os.getenv("TRITON_MODEL_NAME", "flood_model")

_mask_array = np.load(MASK_PATH)

def _get_client():
    return httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

def _infer_and_geojson(rain_feat, batch_size: int) -> str:
    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(f"{FEATURE_PATH} not found")
    
    client = _get_client()

    if not client.is_server_ready():
        raise RuntimeError(f"Triton not ready at {TRITON_SERVER_URL}")
    try:
        if not client.is_model_ready(TRITON_MODEL_NAME):
            raise RuntimeError(f"Model: '{TRITON_MODEL_NAME}' is not ready on Triton")
    except InferenceServerException as e:
        raise RuntimeError(f"Triton model check failed: {e}") from None
    
    try:
        mask = prediction(
            client=client,
            h5_path=FEATURE_PATH,
            rain_feat=rain_feat,
            batch_size=batch_size,
            model_name=TRITON_MODEL_NAME,
        )
    except InferenceServerException as e:
        raise RuntimeError(f"Triton inference failed: {e}") from None
    
    mask[_mask_array == 1] = 0
    save_flood_mask_png(mask)
    extract_flooded_links(result=mask, transform=TRANSFORM, crs=CRS)
    return result_to_geojson(mask, TRANSFORM, CRS)


@shared_task(name="simulation_task")
def simulation_task(rain: list[float], batch_size: int) -> str:
    rain_feat = rain_variables(rain)
    return _infer_and_geojson(rain_feat, batch_size)

@shared_task(name="realtime_task")
def realtime_task(forecast_hour: int, batch_size: int) -> str:
    rain = extract_rain_in_api(forecast_hour)
    rain_feat = rain_variables(rain)
    return _infer_and_geojson(rain_feat, batch_size)