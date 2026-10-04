"""Competition gateway adapter; all decisions remain in the V5 controller."""

import os
from urllib.parse import urlparse

from agents.yf_arc3_v5.agent import YFArc3V5Agent


class MyAgent(YFArc3V5Agent):
    def __init__(self, *args, arc_env=None, **kwargs):
        if arc_env is not None:
            raise RuntimeError(
                "This bundle uses the official gateway HTTP profile. "
                "The mounted arc_env framework needs notebook integration first."
            )
        if os.environ.get("YF_ARC3_V5_MANUAL_PLAY", "").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            raise RuntimeError("Human manual-play mode is unavailable in submission inference")
        root_url = kwargs.get("ROOT_URL", args[3] if len(args) > 3 else "")
        if kwargs.get("controller") is None and kwargs.get("transport") is None:
            parsed = urlparse(root_url)
            if (parsed.scheme, parsed.hostname, parsed.port) != ("http", "gateway", 8001):
                raise ValueError("Submission HTTP transport must use http://gateway:8001")
        if kwargs.get("record", args[4] if len(args) > 4 else False):
            raise ValueError("Submission-side recording must be disabled")
        super().__init__(*args, **kwargs)
