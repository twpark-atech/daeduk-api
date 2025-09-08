# api.py

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from typing import List, Literal

from celery.result import AsyncResult
from .celery_app import celery_app
from .tasks import simulation_task, realtime_task

app = FastAPI(title="Flood Prediction Celery API")

class SimulationRequest(BaseModel):
    rain: List[float]
    batch_size: int = 64
    mode: Literal["async", "sync"] = "async"

class RealtimeRequest(BaseModel):
    forecast_hour: int
    batch_size: int = 64
    mode: Literal["async", "sync"] = "async"

@app.post("/simulation")
def simulation(req: SimulationRequest):
    if req.mode == "sync":
        try:
            geojson_str = simulation_task.run(req.rain, req.batch_size)
            return Response(content=geojson_str, media_type="application/json")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    task = simulation_task.delay(req.rain, req.batch_size)
    return {"task_id": task.id, "status": "PENDING"}

@app.post("/realtime")
def realtime(req: RealtimeRequest):
    if req.mode == "sync":
        try:
            geojson_str = realtime_task.run(req.forecast_hour, req.batch_size)
            return Response(content=geojson_str, media_type="application/json")
        except FileNotFoundError as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    task = realtime_task.delay(req.forecast_hour, req.batch_size)
    return {"task_id": task.id, "status": "PENDING"}

@app.get("/tasks/{task_id}")
def task_status(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    payload = {"task_id": task_id, "state": res.state}
    if res.state == "PENDING":
        payload["progress"] = 0
    if res.state == "STARTED":
        payload["progress"] = 10
    if res.state == "SUCCESS":
        payload["progress"] = 100
    if res.state == "FAILURE":
        payload["error"] = str(res.info)
    return JSONResponse(payload)

@app.get("/tasks/{task_id}/result")
def task_result(task_id: str):
    res = AsyncResult(task_id, app=celery_app)
    if not res.ready():
        return JSONResponse({"task_id": task_id, "state": res.state}, status_code=202)
    if res.failed():
        raise HTTPException(status_code=500, detail=str(res.info))
    geojson_str: str = res.get()
    return Response(content=geojson_str, media_type="application/json")