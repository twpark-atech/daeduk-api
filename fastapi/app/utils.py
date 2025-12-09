import os
import uuid
import re
import json
import h5py
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from rasterio import features
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.ops import unary_union
from shapely.geometry import shape
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from typing import Dict, Any
from lxml import etree
from dateutil.tz import gettz
from datetime import datetime, timezone
from openai import AzureOpenAI
from dotenv import load_dotenv

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
    if result.ndim != 2:
        raise ValueError(f"result_to_geojson expects 2D array, got shape={result.shape}")

    h, w = result.shape
    
    if BLOCK_SIZE > 1:
        h_trim = (h // BLOCK_SIZE) * BLOCK_SIZE
        w_trim = (w // BLOCK_SIZE) * BLOCK_SIZE
        result_trim = result[:h_trim, :w_trim]

        result_blocks = result_trim.reshape(
            h_trim // BLOCK_SIZE, BLOCK_SIZE,
            w_trim // BLOCK_SIZE, BLOCK_SIZE
        )
        result_coarse = result_blocks.max(axis=(1, 3))

        a = transform.a * BLOCK_SIZE
        b = transform.b * BLOCK_SIZE
        c = transform.c
        d = transform.d * BLOCK_SIZE
        e = transform.e * BLOCK_SIZE
        f = transform.f
        transform_coarse = type(transform)(a, b, c, d, e, f)
    else:
        result_coarse = result
        transform_coarse = transform

    if USE_GAUSSIAN:
        result_smooth = gaussian_filter(result_coarse, sigma=SIGMA)
    else:
        result_smooth = result_coarse

    mask = result_smooth >= MASK_THRESHOLD

    if np.count_nonzero(mask) == 0:
        empty_gdf = gpd.GeoDataFrame(
            {"depth": []},
            geometry=[],
            crs=crs
        ).to_crs("EPSG:4326")
        ouput_path = os.path.join(RPATH, "prediction_geojson")
        empty_gdf.to_file(ouput_path)
        return empty_gdf.to_json()

    shapes_gen = features.shapes(
        mask.astype(np.uint8),
        mask=(mask),
        transform=transform_coarse
    )
    
    polygons = []
    mean_depths = []

    for geom, val in shapes_gen:
        if val == 1:  # 침수 지역만
            poly = shape(geom)

            # polygon 영역에 해당하는 픽셀만 추출
            mask_indices = features.geometry_mask([geom], result_smooth.shape, transform_coarse, invert=True)
            depths = result_smooth[mask_indices]

            mean_depth = float(depths.mean()) if depths.size > 0 else 0.0
            polygons.append(poly)
            mean_depths.append(round(mean_depth, 2))

    # GeoDataFrame으로 변환 후 EPSG:4326 좌표계로 변환
    flood_gdf = gpd.GeoDataFrame(
        {"depth": mean_depths},
        geometry=polygons,
        crs=crs
    ).to_crs("EPSG:4326")

    ouput_path = os.path.join(RPATH, "prediction_geojson")
    flood_gdf.to_file(ouput_path)
    geojson_str = flood_gdf.to_json()
    return geojson_str

def save_flood_mask_png(result):
    if result.ndim != 2:
        raise ValueError(f"result_to_png expects 2D array, got shape={result.shape}")

    mask = result >= MASK_THRESHOLD
    h, w = result.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    r, g, b = COLOR

    rgba[mask, 0] = r
    rgba[mask, 1] = g
    rgba[mask, 2] = b
    rgba[mask, 3] = ALPHA

    img = Image.fromarray(rgba, mode="RGBA")
    img.save(os.path.join(RPATH, 'prediction.png'), format="PNG")

def extract_flooded_links(result, transform, crs):
    if result.ndim != 2:
        raise ValueError(f"result_to_png expects 2D array, got shape={result.shape}")
    
    mask = result >= MASK_THRESHOLD
    h, w = result.shape

    links = gpd.read_file(LINK_PATH)
    if LINK_ID not in links.columns:
        raise ValueError(f"{LINK_ID} 컬럼이 {LINK_PATH}에 없음.")
    
    if links.crs is None:
        links.set_crs(crs, inplace=True)
    elif links.crs != crs:
        links = links.to_crs(crs)

    links = links.reset_index(drop=True)
    shapes_for_rasterize = [
        (geom, idx + 1)
        for idx, geom in enumerate(links.geometry)
        if geom is not None and not geom.is_empty
    ]

    link_grid = rasterize(
        shapes=shapes_for_rasterize,
        out_shape=(h, w),
        transform=transform,
        fill=0,
        dtype="int32"
    )

    flooded_idx = link_grid[mask]
    flooded_idx = flooded_idx[flooded_idx > 0]

    if flooded_idx.size == 0:
        return [], links.iloc[0:0].copy()
    
    unique_idx = np.unique(flooded_idx)

    flooded_links_gdf = links.iloc[unique_idx - 1].copy()
    flooded_idx = flooded_links_gdf[LINK_ID].tolist()

    with open (os.path.join(RPATH, 'flooded_link.txt'), "w", encoding="utf-8") as f:
        for lid in flooded_idx:
            f.write(str(lid) + "\n")

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

def gdf_from_geojson(geojson_str):
    geojson = json.loads(geojson_str)
    if not geojson:
        raise ValueError("GeoJson 없습니다.")
    feats = geojson.get("features")
    records = []
    for f in feats:
        props = (f.get("properties") or {}).copy()
        geom = f.get("geometry")
        if not geom:
            continue
        props["geometry"] = shape(geom)
        records.append(props)
    if not records:
        raise ValueError("유효한 Geometry가 없습니다.")
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf

def build_clusters_meter(gdf_m,
                         gap_m=40, simplify_m=20, min_area_m2=2_000,
                         dbscan_eps_m=150, dbscan_min_samples=1):
    gdfb = gdf_m.copy()
    gdfb["geometry"] = gdfb.buffer(gap_m)
    merged = unary_union(gdfb.geometry)
    parts = list(getattr(merged, "geoms", [merged]))
    clusters_m = gpd.GeoDataFrame({"geometry": parts}, crs=gdf_m.crs)
    clusters_m["geometry"] = clusters_m.buffer(-gap_m).buffer(0)

    clusters_m["area_m2"] = clusters_m.area
    clusters_m = clusters_m[clusters_m["area_m2"] >= min_area_m2].copy()
    if len(clusters_m) == 0:
        return clusters_m
    clusters_m["geometry"] = clusters_m.simplify(simplify_m, preserve_topology=True)

    centers = np.array([[g.centroid.x, g.centroid.y] for g in clusters_m.geometry])
    labels = DBSCAN(eps=dbscan_eps_m, min_samples=dbscan_min_samples).fit_predict(centers)
    clusters_m["cluster"] = labels
    clusters_m = clusters_m.dissolve(by="cluster", as_index=False)
    return clusters_m

def circles_from_polygons(clusters_m, pad_m=80):
    out = []
    clusters_wgs = clusters_m.to_crs(4326)
    for geom_m, geom_wgs in zip(clusters_m.geometry, clusters_wgs.geometry):
        if geom_m.geom_type == "MultiPolygon":
            geom_m = max(geom_m.geoms, key=lambda p: p.area)
        c_m = geom_m.centroid
        coords = np.asarray(geom_m.exterior.coords)
        r_m = np.max(np.sqrt(((coords - np.array([c_m.x, c_m.y]))**2).sum(axis=1))) + pad_m
        c_wgs = gpd.GeoSeries([c_m], crs=clusters_m.crs).to_crs(4326).iloc[0]
        cap_circle = f"{c_wgs.y:.6f},{c_wgs.x:.6f} {r_m/1000.0:.3f}"
        out.append({"geometry": geom_wgs, "cap_circle": cap_circle, "radius_m": r_m})
    return gpd.GeoDataFrame(out, crs=4326)

def auto_reduce_to_max_areas_with_depth(gdf_wgs, max_areas=15):
    """
    영역 수가 max_areas 이하가 될 때까지 축약하고,
    circles 테이블에 depth 통계(평균/최대/최소, 미터)를 붙여 반환.
    """
    gdf_m = gdf_wgs.to_crs(3857).copy()

    # depth 컬럼 정규화: depth_m 우선, 없으면 depth_cm→미터 변환
    if "depth" in gdf_m.columns:
        gdf_m["_depth"] = gdf_m["depth"].astype(float)
    elif "depth_cm" in gdf_m.columns:
        gdf_m["_depth"] = gdf_m["depth_cm"].astype(float) / 100.0
    else:
        gdf_m["_depth"] = np.nan  # 깊이 데이터 없음

    gap_m, eps_m = 40, 150
    simplify_m, min_area_m2 = 20, 2_000
    pad_m = 80

    clusters_m = None
    for _ in range(20):
        clusters_m = build_clusters_meter(
            gdf_m,
            gap_m=gap_m,
            simplify_m=simplify_m,
            min_area_m2=min_area_m2,
            dbscan_eps_m=eps_m,
            dbscan_min_samples=1
        )
        n = len(clusters_m)
        if n == 0:
            min_area_m2 = max(500, int(min_area_m2*0.5))
            simplify_m   = max(5,   int(simplify_m*0.5))
            gap_m        = max(10,  int(gap_m*0.5))
            eps_m        = max(80,  int(eps_m*0.5))
            continue

        if n <= max_areas:
            break

        gap_m       = int(gap_m * 1.4)
        eps_m       = int(eps_m * 1.4)
        min_area_m2 = int(min_area_m2 * 1.6)
        simplify_m  = int(simplify_m * 1.2)
        pad_m       = int(pad_m * 1.1)

    # circles 생성
    circles = circles_from_polygons(clusters_m, pad_m=pad_m)  # geometry=WGS84

    # --- 깊이 통계 붙이기 ---
    # 공간조인: 원본 셀(미터 좌표) vs 최종 폴리곤(미터 좌표)
    clusters_m = clusters_m.reset_index(drop=True)
    clusters_m["cluster_id"] = clusters_m.index  # 조인 키
    joined = gpd.sjoin(
        gdf_m[["_depth", "geometry"]].dropna(subset=["_depth"]),
        clusters_m[["cluster_id", "geometry"]],
        how="inner",
        predicate="intersects"
    )

    if len(joined) > 0:
        stats = joined.groupby("cluster_id")["_depth"].agg(
            depth_mean_m="mean",
            depth_min_m="min",
            depth_max_m="max",
            n_cells="count",
        ).reset_index()
        # circles와 cluster_id를 맞추기 위해 같은 순서의 인덱스 사용
        # circles는 clusters_m 순서를 따라 만들었으므로 인덱스==cluster_id 가 동일
        stats = stats.set_index("cluster_id").reindex(clusters_m["cluster_id"]).reset_index()
        circles["depth_mean_m"] = stats["depth_mean_m"].values
        circles["depth_min_m"]  = stats["depth_min_m"].values
        circles["depth_max_m"]  = stats["depth_max_m"].values
        circles["n_cells"]      = stats["n_cells"].values
    else:
        circles["depth_mean_m"] = np.nan
        circles["depth_min_m"]  = np.nan
        circles["depth_max_m"]  = np.nan
        circles["n_cells"]      = 0

    # (선택) cm 단위 컬럼도 같이
    # circles["depth_mean_cm"] = circles["depth_mean_m"] * 100.0
    # circles["depth_min_cm"]  = circles["depth_min_m"]  * 100.0
    # circles["depth_max_cm"]  = circles["depth_max_m"]  * 100.0

    # Serverity Assign
    d = circles["depth_mean_m"]
    circles["severity"] = np.select(
        [
            d < 0.3,
            (d >= 0.3) & (d < 0.5),
            (d >= 0.5) & (d < 1.0),
            d >= 1.0
        ],
        ["Minor", "Moderate", "Severe", "Extreme"],
        default="Unknown"   # 결측 등
    )

    # 최종 폴리곤(4326), 원(4326+통계) 반환
    return clusters_m.to_crs(4326), circles

def _pick(addr, keys):
    for k in keys:
        v = addr.get(k)
        if v: 
            return v
    return None

def reverse_kr(lat: float, lon: float, lang: str = "ko", user_agent="demo_app"):
    geolocator = Nominatim(user_agent=user_agent)
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1, swallow_exceptions=True)
    loc = reverse((lat, lon), language=lang, addressdetails=True)
    if not loc or not getattr(loc, "raw", None):
        return None, None, None, None, None, None  # 실패

    addr = loc.raw.get("address", {})
    country_code = (addr.get("country_code") or "").upper()

    # 공통(전세계용) 기본 우선순위
    keys_country  = ["country"]
    keys_postcode = ["postcode"]
    # KR 특화 우선순위 (시/도, 시/군/구, 읍/면/동/리)
    # 시/도
    keys_si_do    = ["state", "province", "region"]  # 예: 서울특별시, 경상남도
    # 시/군/구
    keys_si_gun_gu = ["city", "town", "county", "district", "municipality", "city_district", "borough"]
    # 읍/면/동/리
    keys_eup_myeon_dong = ["suburb", "neighbourhood", "village", "town", "hamlet", "quarter", "residential"]

    # 추출
    si_do   = _pick(addr, keys_si_do)
    sigungu = _pick(addr, keys_si_gun_gu)
    dong    = _pick(addr, keys_eup_myeon_dong)
    postcode= _pick(addr, keys_postcode)
    country = _pick(addr, keys_country)

    # 중복 제거(가끔 동일 값이 다른 키로 중복 표기됨)
    def _dedup(seq):
        seen, out = set(), []
        for s in seq:
            if s and s not in seen:
                seen.add(s); out.append(s)
        return out

    # 요청하신 출력 순서: 동 / 구(군) / 시(도) / 국가  (우편번호는 옵션)
    parts_no_zip  = _dedup([si_do, sigungu, dong])
    parts_with_zip= _dedup([si_do, sigungu, dong, postcode])

    label_no_zip  = " ".join(p for p in parts_no_zip if p)
    label_with_zip= " ".join(p for p in parts_with_zip if p)

    return dong, sigungu, si_do, postcode, country, (label_no_zip, label_with_zip)

def composition_data(gdf):
    clusters_4326, circles = auto_reduce_to_max_areas_with_depth(gdf)
    for i in circles.index:
        lat=float(circles['cap_circle'][i].split(' ')[0].split(',')[0])
        lon=float(circles['cap_circle'][i].split(' ')[0].split(',')[1])
        dong, sigungu, sido, postcode, country, (addr_no_zip, addr_with_zip) = reverse_kr(lat,lon)
        circles.loc[i,'address'] = addr_no_zip
    return circles

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

KST = gettz("Asia/Seoul")
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")

def _client():
    if not ENDPOINT or not API_KEY:
        raise RuntimeError("Azure OpenAI credentials not configured. Check .env")
    return AzureOpenAI(
        api_key=API_KEY,
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
    )

def _chat(messages, max_tokens=800):
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    print("LLM response:", resp)
    return resp.choices[0].message.content

def translate_text(text: str, target_lang: str) -> str:
    langdict = {"en": "English", "vi": "Vietnamese", "id": "Indonesian", "ko": "Korean"}
    if target_lang not in langdict:
        raise ValueError(f"Unsupported target_lang: {target_lang}")
    target_lang_fullname = langdict[target_lang]
    system = "You are a precise translator. Preserve numbers and named entities. Keep it short."
    prompt = f"Translate the following text to {target_lang_fullname}.\n\nTEXT:\n{text}"
    return _chat([{"role": "system", "content": system},
                  {"role": "user", "content": prompt}], max_tokens=400)


def cap_to_announcer_script(cap: Dict[str, Any], target_lang:str="ko") -> str:
    info = cap.get("info", {})
    print(info)
    prompt = f"""
You are a professional broadcast announcer. Convert the following CAP (Common Alerting Protocol) fields
into a concise, calm, and authoritative live-read script in the same language specified by `language` as colloquial as possible.
Use short sentences, clear actions for the public, and include the location, event, severity, and effective time.
Do not insert "boradcast start" or "end of brodcast" phrases. 
Do not mention of certainty of CAP, CAP fields and actual level of severity.

### Example 1
CAP FIELDS:
language: ko
event: 홍수
location: 서울 강남구
severity: Extreme
certainty: Observed
effective: 2025-07-21T06:30
instructions:
- 저지대 주민 대피
- 차량 운행 자제
- 추가 안내 확인

OUTPUT:
서울 강남구 일원에서 홍수 상황이 관측되었습니다.
현재 일부 저지대 도로와 주택가에 물이 차오르고 있어 주민 여러분의 각별한 주의가 필요합니다.

- 주민 여러분께서는 즉시 안전한 곳으로 이동하시기 바랍니다.  
- 차량 운행을 자제해 주시고, 저지대 접근을 삼가 주십시오.  
- 앞으로의 추가 안내와 지침은 관계 기관을 통해 수시로 확인해 주시기 바랍니다.

시민 여러분의 안전이 최우선입니다. 
침착하게 행동하시고, 지역 당국의 안내에 적극 협조해 주시기 바랍니다.

---

### Example 2
CAP FIELDS:
language: ko
event: 집중호우
location: 대전 유성구
severity: Moderate
certainty: Observed
effective: 2025-08-02T17:45
instructions:
- 차량 이동
- 지하 접근 금지
- 구조대 협조

OUTPUT:
대전 유성구 지역에 집중호우가 관측되었습니다.
현재 일부 아파트 단지 지하주차장으로 빗물이 유입되고 있어, 주민 여러분의 주의가 필요합니다.

- 차량을 지하주차장에서 즉시 이동시켜 주시기 바랍니다.
- 지하 공간 출입은 매우 위험하므로, 지하 접근을 삼가해 주십시오.
- 구조대 및 관계 당국의 안내가 있을 경우, 신속히 협조해 주시기 바랍니다.

시민 여러분, 작은 부주의가 큰 피해로 이어질 수 있습니다.
지속적으로 안내되는 기상 상황과 안전 지침을 확인하시고, 안전 확보에 각별히 유의해 주시기 바랍니다.

---

### Example 3
CAP FIELDS:
language: ko
event: 침수
location: 부산 연제구 연산동
severity: Severe
certainty: Observed
effective: 2025-08-27T09:09
instructions:
- 주민 대피
- 침수 지역 회피
- 차량 운행 자제
- 오염 주의

OUTPUT:
부산 연제구 연산동 일대에서 심각한 침수 상황이 관측되었습니다.
현재 일부 주택과 도로가 물에 잠겼으며, 하수 역류로 인한 오염 우려도 제기되고 있습니다.

- 주민 여러분께서는 즉시 안전한 곳으로 대피하시기 바랍니다.
- 침수 지역과 주변 도로는 접근을 피하시고, 차량 운행은 삼가 주십시오.
- 빗물에 섞인 오염물질에 접촉하지 않도록 각별히 주의해 주시기 바랍니다.

추가 안내는 관계 당국을 통해 전달될 예정입니다.
시민 여러분의 안전이 최우선입니다. 침착하게 행동하시고, 당국의 지시에 적극 협조해 주시기 바랍니다.

### Example 4
CAP FIELDS:
language: ko
event: 집중호우
location: 울산 중구 신정동
severity: minor
certainty: Likely
effective: 2025-06-22T06:30
instructions:
- 우산 및 우비 등 개인 대비 용품 준비
- 배수구, 하수구 주변에 쓰레기나 낙엽 치우기
- 차량은 안전한 장소에 주차
- 기상 특보 및 추가 안내 지속 확인

OUTPUT:
울산 중구 신정동 지역에 집중호우 가능성이 예보되었습니다.
현재까지는 소규모 피해 가능성으로 분류되고 있으나, 시민 여러분의 각별한 주의가 필요합니다.

- 외출 시 우산, 우비 등 개인 대비 용품을 준비해 주시기 바랍니다.
- 집 주변의 배수구와 하수구는 미리 정리하여 침수를 예방해 주십시오.
- 차량은 지대가 낮은 곳을 피해 안전한 장소에 주차하시기 바랍니다.
- 앞으로의 기상 특보와 추가 안내는 지속적으로 확인해 주시기 바랍니다.

안전을 위한 작은 준비가 큰 피해를 막습니다. 시민 여러분의 주의와 협조를 부탁드립니다.

---

## Task
Now, generate the announcer script for the following CAP FIELDS:
{info}

Format:
- 2-3 sentence intro: what/where/when.
- 2-4 bullet-style lines (spoken) of actions to take.
- End with a reassurance/monitoring line.

Keep it under 120 words (or 300 Korean characters).
"""
    script= _chat([{"role": "user", "content": prompt}], max_tokens=500)
    if target_lang != "ko":
        script = translate_text(script, target_lang)
    return script

def _to_hhmm_from_any(s: str) -> str | None:
    """'2024-06-22 23:31', '2024-06-22T23:31:45+09:00', '23:31' 등에서 HH:MM 추출"""
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{2}:\d{2}", s):
        return s
    try:
        dt = dateparser.parse(s)
        return dt.strftime("%H:%M")
    except Exception:
        return None
    
def cap_to_sms(
        cap: Dict[str, Any], 
        target_lang: str = "ko", 
        max_chars: int = 120, 
        time_hint: str | None = None, 
        style: str = "advisory",
        url: str | None = None,
        department: str | None = None, 
        ) -> str:
    langdict = {"en": "English", "vi": "Vietnamese", "id": "Indonesian", "ko": "Korean"}
    if target_lang not in langdict:
        raise ValueError(f"Unsupported target_lang: {target_lang}")
    target_lang_fullname = langdict[target_lang]
    info = cap.get("info", {})
    cap_sent = cap.get("sent") 
    hhmm = _to_hhmm_from_any(time_hint) or _to_hhmm_from_any(cap_sent)
    system = f"""
    You compress CAP alerts into safety SMS. 
    Use no emojis. 
    Prioritize location, event, action. 
    Keep within a character limit and target language.
    Generate a SMS with style of {style}.

    #Additional Guidelines:
    You compress CAP alerts into neutral public-safety SMS.
    - Do not include graphic or violent descriptions.
    - Use clear, calm, non-alarming wording.
    - No emojis, no exclamation marks.
    - Keep it short; focus on location, event, actionable guidance.
    - Treat "중동" here as the neighborhood "Jung-dong" as a Korean place name, NOT the Middle East. Do not translate or generalize it.
    - Do not mention war, crime, or violent conflict. Use neutral public-safety wording only.
    - When you use example #4 case, do not insert the future plans.
    - Do not mention "CAP", "Common Alerting Protocol", or any CAP fields in the brackets. such as [현장 제보]

    ## Examples
    ### Example 1 (advisory) - 친화형(권고, 설득)
    오늘 저녁 강한 비가 예상됩니다.  
    하천 등 침수 위험이 있는 지역은 잠시 피해 주시고,  
    안전을 위해 각별히 유의해 주시기 바랍니다. [부산 해운대구]  

    특히 침수 예상 지역을 우회할 수 있는  
    안전한 대피 경로를 안내드리오니,  
    안심할 수 있는 곳으로 이동해 주시기 바랍니다.  

    [대피 경로 안내]  
    URL Link

    ### Example 2 (formal) - 공식 통보형(일반)
    오늘 저녁 강한 비가 내릴 것으로 예상되오니,  
    하천 및 저지대 등 침수우려 지역의 통행을 즉시 자제하시기 바랍니다.  
    주민 여러분께서는 안전에 각별히 유의하시고,  

    아래 침수 예상 지역을 회피하여 지정된 대피 경로로 이동하시기 바랍니다. [부산 수영구]  

    [대피 경로 안내]  
    URL Link

    ### Example 3 (alert) - 경고 통보형(의무)
    오늘 저녁 시간대 강한 호우로 침수 피해가 예상됩니다.  
    하천, 지하차도 등 침수위험 지역의 접근은 금지되며,  
    주민 여러분께서는 즉시 안전한 지역으로 대피하시기 바랍니다.  

    아래 안내된 대피 경로를 따라 이동하시기 바랍니다.  [울산 중구] 

    [대피 경로 안내]  
    URL Link
    """

    # ### Example 4 (매우 간결하게 전달 하는 형태)
    # 00:00 ○○시 ○○동 인근에 시간당 ○○mm 이상 강한 비로 침수 등 우려. 안전확보를 위한 국민행동요령 확인 바람 cbs.kma.go.kr [기상청]
    # [비상시 대피 경로 안내]
    # URL Link
    content = f"""
Target language: {target_lang_fullname}
Max characters: {max_chars}

Fields:
event: {info.get('event','')}
area: {info.get('area',{}).get('areaDesc','')}
severity: {info.get('severity','')}
urgency: {info.get('urgency','')}
certainty: {info.get('certainty','')}
headline: {info.get('headline','')}
description: {info.get('description','')}
URL Link: {url if url else 'N/A'}
"""
    body= _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compose the SMS now." + content},
        ],
        max_tokens=160,
    )
    body = body or ""   
    prefix = f"{hhmm} " if hhmm else ""
    tail = f" [{department or '행안부'}]"
    sms = (prefix + body.strip() + tail).strip()

    # if len(sms) > max_chars:
    #     # 여유를 둬서 tail은 반드시 남기고 앞쪽 본문만 자름
    #     keep_tail = len(tail)
    #     keep_prefix = len(prefix)
    #     avail = max_chars - keep_tail - keep_prefix
    #     core = body
    #     if avail < 5:  # 너무 짧으면 그냥 tail만 남는 걸 방지
    #         avail = 5
    #     core = (core[:avail-1] + "…") if len(core) > avail else core
    #     sms = (prefix + core + tail)
    return sms

def _now_iso():
    # 기존: UTC → 변경: KST
    return datetime.now(KST).replace(microsecond=0).isoformat()

def _text(parent, tag, val):
    el = etree.SubElement(parent, tag)
    el.text = str(val)
    return el

def cap_form_model(data, target_lang:str="ko"):
    langdict = {"en": "English", "vi": "Vietnamese", "id": "Indonesian", "ko": "Korean"}
    if target_lang not in langdict:
        raise ValueError(f"Unsupported target_lang: {target_lang}")
    target_lang_fullname = langdict[target_lang]
    print("langauge checkers : ", target_lang_fullname)
    sender = os.getenv("CAP_SENDER")
    identifier = f"cap-{uuid.uuid4()}"
    # polygons = geom_to_cap_polygons(data['geometry'])
    # centroid = 
    area_desc = data['address']
    event = "flood"
    severity = data['severity']
    certainty = "Likely"
    urgency = "Expected"
    language = target_lang_fullname
    effective = _now_iso()
    sent = _now_iso()
    circle = data['cap_circle']
    system = f"""
    You generate the description of disaster in a CAP form.
    # Guide Line 
    - Make a description in short sentence.
    - do not mention about the any inputs.
    - Generate the description with the given language
    - Do not mention of certainty of CAP, CAP fields and actual level of severity.
    # Note
    - Do not use harmful or exaggerated expression of given disaster.
    - Treat "중동" here as the neighborhood "Jung-dong" as a Korean place name, NOT the Middle East. Do not translate or generalize it.
    - Do not mention war, crime, or violent conflict. Use neutral public-safety wording only.
    - Do not mention "CAP", "Common Alerting Protocol", or any CAP fields in the brackets. such as [현장 제보]
    # Example
    ## Example 1
    울산 중구 태화시장 일대 침수 피해 발생
    ## Example 2
    울산 울주군 언양읍 일대 밤사이 332mm 집중호우 예정
    ## Example 3
    부산광역시 해운대구 중동 일대 침수피해 발생
    ## Example 4
    부산광역시 연제구 연산동 일원 하천범람 및 내수 침수 발생
    ## Example 5
    울산시 울주군 삼동명 침수 발생 위험도 높음
""" 
    content=f"""
# Field :
    area : {area_desc}
    event : {event}
    certainty : {certainty}
    urgency : {urgency}
    severity : {severity}
    language : {target_lang_fullname}
"""
    description= _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Compose the SMS now." + content},
        ],
        max_tokens=160,
    )
    expires = None  # optional
    alert = etree.Element("alert")
    _text(alert, "identifier", identifier)
    _text(alert, "sender", sender)
    _text(alert, "sent", sent)
    _text(alert, "status", "Actual")
    _text(alert, "msgType", "Alert")
    _text(alert, "scope", "Public")

    info_el = etree.SubElement(alert, "info")
    _text(info_el, "language", language)
    _text(info_el, "category", "Met")
    _text(info_el, "event", event)
    _text(info_el, "urgency", urgency)
    _text(info_el, "severity", severity)
    _text(info_el, "certainty", certainty)
    _text(info_el, "description", description)
    _text(info_el, "effective", effective)

    if expires:
        try:
            _text(info_el, "expires", dateparser.parse(expires).astimezone(timezone.utc).isoformat())
        except Exception:
            _text(info_el, "expires", expires)

    area_el = etree.SubElement(info_el, "area")
    _text(area_el, "areaDesc", area_desc)
    _text(area_el, "circle", circle)

    return etree.tostring(alert, encoding="unicode", pretty_print=True)

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