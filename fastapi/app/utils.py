import os
import json
import h5py
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio import features
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely.geometry import shape
from scipy.interpolate import interp1d
from datetime import datetime, timedelta

import tritonclient.http as httpclient
from fastapi import FastAPI, HTTPException
from .config import *

def interpolate_rain(hourly_rain):
    x = np.arange(len(hourly_rain))
    xq = np.linspace(0, len(hourly_rain)-1, num=(len(hourly_rain)-1)*6 + 1)
    return interp1d(x, hourly_rain, kind='linear')(xq)

def make_rain_variables2(rainlist):
    r = np.array(rainlist, dtype='float32')
    dur = float(len(r))
    depth = np.sum(r)
    indpeak = np.argmax(r)
    rcg = (np.min(np.where(np.cumsum(r) >= depth / 2)) + 1) / dur
    rp = (indpeak + 1) / dur
    m1 = np.sum(r[:indpeak]) / np.sum(r[indpeak:]) if np.sum(r[indpeak:]) > 0 else 0
    m2 = np.max(r) / depth if depth > 0 else 0
    m3 = np.sum(r[:int(dur // 3)]) / depth if depth > 0 else 0
    m5 = np.sum(r[:int(dur // 2)]) / depth if depth > 0 else 0
    ni = np.max(r) / np.mean(r) if np.mean(r) > 0 else 0
    return [dur, depth, rp, rcg, m1, m2, m3, m5, ni]

def normalize_rain_variables(rain_feat):
    rmin = [3.0, 0.2111, 0.0270, 0.0789, 0.0000, 0.0491, 0.0062, 0.0249, 1.5218]
    rrange = [204.0, 4.5409, 0.9105, 0.8586, 1.9274, 0.8309, 0.9777, 0.9693, 38.5254]
    return [(x - a) / b if b > 0 else 0 for x, a, b in zip(rain_feat, rmin, rrange)]

def rain_variables(rain):
    rain_interp = interpolate_rain(rain)
    rain_norm = [x / 25.0 for x in rain_interp]
    rain_feat = make_rain_variables2(rain_norm)
    rain_feat_norm = normalize_rain_variables(rain_feat)
    input_2 = np.array(rain_feat_norm, dtype=np.float32)
    return input_2

def parse_rn1(fcst_value):
    if isinstance(fcst_value, str):
        if '강수없음' in fcst_value:
            return 0.0
        elif '1mm 미만' in fcst_value:
            return 0.5
        elif '30.0~50.0mm' in fcst_value:
            return 40.0
        elif '50.0mm 이상' in fcst_value:
            return 50.0
        elif 'mm' in fcst_value and '~' not in fcst_value:
            return float(fcst_value.replace('mm', ''))
        else:
            return 0.0
    else:
        return float(fcst_value)

def prediction(client, h5_path, rain_feat, batch_size=1, model_name: str = "flood_model"):
    pad_h = (PATCH_SIZE - ORIGINAL_H % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - ORIGINAL_W % PATCH_SIZE) % PATCH_SIZE
    padded_h = ORIGINAL_H + pad_h
    padded_w = ORIGINAL_W + pad_w
    num_patch_h = padded_h // PATCH_SIZE
    num_patch_w = padded_w // PATCH_SIZE
    result = np.zeros((padded_h, padded_w), dtype=np.float32)
    with h5py.File(h5_path, "r") as hf:
        dset = hf['features']

        for i in range(0, dset.shape[0], batch_size):
            batch_indices = list(range(i, min(i + batch_size, dset.shape[0])))
            top_feat_batch = []

            for patch_idx in batch_indices:
                patch = dset[patch_idx]
                top = patch.transpose(1, 2, 0).astype(np.float32).copy()
                top_feat_batch.append(top)

            top_feat = np.stack(top_feat_batch, axis=0).astype(np.float32)
            rain_batch = np.broadcast_to(rain_feat, (len(top_feat), rain_feat.shape[0]))

            inputs = [
                httpclient.InferInput("input_1", top_feat.shape, "FP32"),
                httpclient.InferInput("input_2", rain_batch.shape, "FP32"),
            ]
            inputs[0].set_data_from_numpy(top_feat)
            inputs[1].set_data_from_numpy(rain_batch)
            outputs = [httpclient.InferRequestedOutput("reshape_1")]

            response = client.infer(model_name=model_name, inputs=inputs, outputs=outputs)
            output = response.as_numpy("reshape_1")

            if output.shape[1:] != (PATCH_SIZE, PATCH_SIZE, 1):
                raise HTTPException(status_code=500, detail=f"Unexpected output shape: {output.shape}")

            for b, batch_idx in enumerate(batch_indices):
                row = (batch_idx // num_patch_w) * PATCH_SIZE
                col = (batch_idx % num_patch_w) * PATCH_SIZE
                result[row:row+PATCH_SIZE, col:col+PATCH_SIZE] = output[b, :, :, 0]

    result = result[0:ORIGINAL_H, 0:ORIGINAL_W]
    return result

def result_to_geojson(result, transform, crs):
    mask = result >= MASK_THRESHOLD
    shapes_gen = features.shapes(
        mask.astype(np.uint8),
        mask=(mask >= MASK_THRESHOLD),
        transform=transform
    )
    
    polygons = []
    mean_depths = []

    for geom, val in shapes_gen:
        if val == 1:  # 침수 지역만
            poly = shape(geom)

            # polygon 영역에 해당하는 픽셀만 추출
            mask_indices = features.geometry_mask([geom], result.shape, transform, invert=True)
            depths = result[mask_indices]

            mean_depth = float(depths.mean()) if depths.size > 0 else 0.0
            polygons.append(poly)
            mean_depths.append(round(mean_depth, 2))

    # GeoDataFrame으로 변환 후 EPSG:4326 좌표계로 변환
    flood_gdf = gpd.GeoDataFrame(
        {"depth": mean_depths},
        geometry=polygons,
        crs=crs
    ).to_crs("EPSG:4326")

    flood_gdf.to_file(os.path.join(RPATH, 'prediction.geojson'))
    geojson_str = flood_gdf.to_json()
    return geojson_str

def get_before_rain(target_hour, base_time):
    tm2 = base_time + "00"
    tm1 = (datetime.strptime(tm2, "%Y%m%d%H%M") - timedelta(hours=target_hour)).strftime("%Y%m%d%H") + "00"
    params = {
        "tm1": tm1,
        "tm2": tm2,
        "stn": 159,
        "authKey": KMA_SFCTM3_KEY
    }
    try:
        res = requests.get(KMA_SFCTM3_URL, params=params)
        res.raise_for_status()
        text = res.text
    except Exception as e:
        print(f"[종관기상관측 API 요청 실패: {e}]")
        return [0] * target_hour
    start = text.find("#START7777")
    end = text.find("#7777END")
    if start == -1 or end == -1:
        print("[종관기상관측 강수 데이터 없음]")
        return [0] * target_hour
    data_lines = [
        line for line in text[start:end].splitlines() if line.strip() and line[0].isdigit()
    ]
    rain_list = []
    for line in data_lines:
        try:
            val = float(line.split()[15])
            rain_list.append(0.0 if val == -9.0 else val)
        except:
            rain_list.append(0.0)
    rain_arr = np.array(rain_list[-target_hour:], dtype='float32')
    return rain_arr.tolist()

def get_after_rain(target_hour, base_time):
    params = {
        'serviceKey': GETULTRASRTFCST_KEY,
        'pageNo': '1',
        'numOfRows': '1000',
        'dataType': 'JSON',
        'base_date': base_time[:-2],
        'base_time': base_time[-2:] + "00",
        'nx': '98',
        'ny': '76'
    }
    try:
        res = requests.get(GETULTRASRTFCST_URL, params=params)
        res.raise_for_status()
        items = json.loads(res.text)['response']['body']['items']['item']
    except Exception as e:
        print(f"[단기예보 API 요청 실패: {e}]")
        return [0] * target_hour
    rain_data = [
        parse_rn1(item['fcstValue'])
        for item in items if item['category'] == 'RN1'
    ]
    return rain_data[:target_hour]

def extract_rain_in_api(after_time):
    before_time = 34 - after_time
    now = datetime.now()
    if now.minute < 30:
        base_time = (now - timedelta(hours=1)).strftime("%Y%m%d%H")
    else:
        base_time = now.strftime("%Y%m%d%H")
    before_rain = get_before_rain(before_time, base_time)
    after_rain = get_after_rain(after_time, base_time)
    return before_rain + after_rain