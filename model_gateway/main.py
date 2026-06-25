"""Compatibility entrypoint for the FastAPI model proxy server."""

from model_gateway.app import create_app

__all__ = ["create_app"]
