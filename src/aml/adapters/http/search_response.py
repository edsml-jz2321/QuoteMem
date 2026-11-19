"""The Search HTTP projection and its exact encoded bytes (V1 only)."""

from aml.adapters.http.models import SearchHitV1, SearchResponseV1
from aml.api.bench import SearchResult


def encode_search_response(result: SearchResult) -> bytes:
    """Encode ``{data: [...]}`` in rank order, omitting absent annotations.

    Args:
        result: Complete application result already authorized by its caller.

    Returns:
        Exact UTF-8 public response bytes. Serialization grants no delivery authority.
    """
    return SearchResponseV1(
        data=tuple(SearchHitV1(
            id=hit.id, content=hit.content, score=hit.score, created_at=hit.created_at,
        ) for hit in result.hits),
    ).model_dump_json(exclude_none=True).encode("utf-8")
