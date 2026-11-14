"""Pilot route qualification must reject unsupported parameters and price drift."""

import io
import json
from unittest.mock import patch

import pytest

from mcptox_gateway.pipelines.research_pilot import _check_endpoint


def test_endpoint_checks_parameters_and_price_overrides() -> None:
    route = {
        "model": "a/b",
        "provider_slug": "provider",
        "generation": {"max_tokens": 8},
        "max_price": {"prompt": 1, "completion": 2},
        "context_length": 1000,
    }
    endpoint = {
        "tag": "provider",
        "supported_parameters": ["max_tokens", "response_format"],
        "context_length": 1000,
        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
    }

    def response() -> io.BytesIO:
        return io.BytesIO(json.dumps({"data": {"endpoints": [endpoint]}}).encode())

    with patch(
        "mcptox_gateway.pipelines.research_pilot.urlopen", side_effect=lambda *a, **k: response()
    ):
        assert _check_endpoint(route, judge=True)["required_parameters"] == [
            "max_tokens",
            "response_format",
        ]
        endpoint["supported_parameters"] = ["max_tokens"]
        with pytest.raises(ValueError, match="parameters"):
            _check_endpoint(route, judge=True)
        endpoint["supported_parameters"] = ["max_tokens", "response_format"]
        endpoint["pricing"]["overrides"] = [{"prompt": "0.000002", "completion": "0.000002"}]
        with pytest.raises(ValueError, match="price"):
            _check_endpoint(route, judge=True)
