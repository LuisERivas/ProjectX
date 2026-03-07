import os


REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_STREAM_KEY = os.getenv("QUEUE_STREAM_KEY", "jobs:stream")
WORKER_GROUP = os.getenv("WORKER_GROUP", "workers")

