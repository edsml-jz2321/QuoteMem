"""Strict AMLB transport contracts; application behavior remains in the API seam."""

from quotemem.adapters.http.auth import HTTPCredentials
from quotemem.adapters.http.endpoints import PublicEndpoints
from quotemem.adapters.http.errors import HTTPErrorResponse, PublicErrorV1, public_error
from quotemem.adapters.http.models import (
    AddResponseV1,
    SearchHitV1,
    SearchRequestV1,
    SearchResponseV1,
    parse_add_body,
    parse_search_response,
)

__all__ = [
    "AddResponseV1",
    "HTTPCredentials",
    "HTTPErrorResponse",
    "PublicErrorV1",
    "PublicEndpoints",
    "SearchHitV1",
    "SearchRequestV1",
    "SearchResponseV1",
    "parse_add_body",
    "parse_search_response",
    "public_error",
]
