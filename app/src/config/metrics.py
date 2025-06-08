import time

from prometheus_client import Counter, Histogram

PREDICT_REQUEST_COUNT = Counter(
    "predict_request_count", "Total number of predict endpoint requests"
)
PREDICT_SUCCESS_REQUEST_LATENCY = Histogram(
    "predict_request_success_latency_seconds", "Latency (in seconds) for predict endpoint requests"
)

PREDICT_FAILED_REQUEST_LATENCY = Histogram(
    "predict_request_failed_latency_seconds", "Latency (in seconds) for failed (app reason) predict endpoint requests"
)

FAILED_PREDICTION_REQUEST_COUNT = Counter(
    "failed_prediction_request_count", "Total number of unsuccessful prediction requests"
)


def record_duration(metric, start_time):
    duration = time.time() - start_time
    metric.observe(duration)
