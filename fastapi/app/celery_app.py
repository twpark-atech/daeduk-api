# celery_app.py

from __future__ import annotations
import os
from celery import Celery

BROKER = os.environ.get("CELERY_BROKER_URL")
BACKEND = os.environ.get("CELERY_RESULT_BACKEND")

if not BROKER or not BACKEND:
    raise RuntimeError(
        f"Missing broker/backend. CELERY_BROKER_URL={BROKER}, CELERY_RESULT_BACKEND={BACKEND}"
    )
if not (BROKER.startswith("redis://") and BACKEND.startswith("redis://")):
    raise RuntimeError(f"Unexpected broker/backend scheme. broker={BROKER}, backend={BACKEND}")

celery_app = Celery(
    "flood_predict",
    broker=BROKER,
    backend=BACKEND,
    include=["app.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    broker_transport_options={"visibility_timeout": 3600},
    task_queues={
        "inference": {"exchange": "inference", "routing_key": "inference"},
        "celery": {"exchange": "celery", "routing_key": "celery"}
    },
    task_routes={
        "simulation_task": {"queue": "inference", "routing_key": "inference"},
        "realtime_task": {"queue": "inference", "routing_key": "inference"}
    },
)

print("[celery] Broker:", celery_app.conf.broker_url)
print("[celery] Backend:", celery_app.conf.result_backend)