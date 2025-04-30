# AppFiles/metrics.py
from prometheus_client import Counter

# Define all your metrics here
request_count = Counter(
    "request_count",
    "Total number of HTTP requests received"
)