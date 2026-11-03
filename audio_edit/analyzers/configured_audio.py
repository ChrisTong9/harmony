"""Use the configured audio service independently of its network hostname."""

import os
from urllib.parse import urlparse


def compatible_audio_endpoint(backend, endpoint):
    if backend != "openai_compatible":
        return False
    try:
        parsed = urlparse(str(endpoint))
        _ = parsed.port
        return parsed.scheme in {"http", "https"} and bool(parsed.hostname)
    except ValueError:
        return False


def configured_audio_analyzer():
    """Return the existing Qwen service with transport retries disabled."""
    if not compatible_audio_endpoint(
        os.getenv("QWEN_AUDIO_BACKEND"), os.getenv("AUDIO_ANALYSIS_BASE_URL", "")
    ):
        return None
    from audio_edit.analyzers.qwen_audio_analyzer import QwenAudioAnalyzer

    analyzer = QwenAudioAnalyzer()
    if not compatible_audio_endpoint(analyzer.backend, analyzer.client.base_url):
        return None
    analyzer.client = analyzer.client.with_options(max_retries=0)
    return analyzer
