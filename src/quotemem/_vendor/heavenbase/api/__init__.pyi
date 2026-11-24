from heavenbase.utils.typing import Any, Optional, Set

from .service import DashboardAPI as DashboardAPI

def openapi_schema() -> dict[str, Any]: ...
def build_app(
    api: Optional[DashboardAPI] = ...,
    allowed_hosts: Optional[Set[str]] = ...,
    static_dir: Optional[str] = ...,
) -> Any: ...
def resolve_bind(cm: Any = ..., host: Optional[str] = ..., port: Optional[int] = ...) -> tuple[str, int]: ...
def run_server(
    host: Optional[str] = ...,
    port: Optional[int] = ...,
    api: Optional[DashboardAPI] = ...,
    allow_network: bool = ...,
    static_dir: Optional[str] = ...,
) -> None: ...

__all__: list[str]
