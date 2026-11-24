from heavenbase.api import DashboardAPI as _DashboardAPI
from heavenbase.utils.typing import Callable, Literal, Optional
from os import PathLike as _PathLike

HostMode = Literal["auto", "webview", "browser", "app"]

def default_macos_app_path() -> str: ...
def install_macos_app(
    destination: str | _PathLike[str] | None = ...,
    *,
    replace: bool = ...,
    python_executable: str | _PathLike[str] | None = ...,
) -> str: ...
def open_dashboard(url: str, mode: HostMode = ...) -> HostMode: ...
def serve(
    host: Optional[str] = ...,
    port: Optional[int] = ...,
    open_window: Optional[bool] = ...,
    api: Optional[_DashboardAPI] = ...,
    host_mode: HostMode = ...,
    allow_network: bool = ...,
    *,
    on_started: Callable[[str], None] | None = ...,
) -> None: ...

__all__: list[str]
