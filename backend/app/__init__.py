import os

# Model libraries must never reach the network on their own: weights are loaded from the local
# cache only, and the single download path is the explicit `image-pipeline models download` action.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

__version__ = "0.1.0"
PIPELINE_VERSION = "0.1.0"
SCHEMA_VERSION = "1.0"
