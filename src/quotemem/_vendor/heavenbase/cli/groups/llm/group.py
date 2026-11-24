"""LLM CLI command specifications."""

from __future__ import annotations

__all__ = ["llm_group"]

from ...spec import ArgumentSpec, CommandGroupSpec, CommandSpec, OptionSpec
from .catalog import _list, _status, _verbose_cmd
from .chat import _chat
from .embed import _embed
from .image import _imagen
from .runtime import _DEFAULT_MAX_STEPS
from .session import _session


def _llm_options() -> list[OptionSpec]:
    return [
        OptionSpec.from_flags("preset", "--preset", "-p", help="LLM preset."),
        OptionSpec.from_flags("model", "--model", "-m", help="Model key or id."),
        OptionSpec.from_flags("provider", "--provider", "-b", help="Provider key."),
        OptionSpec.from_flags("gateway", "--gateway", "-g", help="Gateway key."),
    ]


def _cache_option() -> OptionSpec:
    return OptionSpec.from_flags("cache", "--cache", "--no-cache", help="Enable cache.", default=True, is_flag=True, type=bool, negatable=True)


def _verbose_option() -> OptionSpec:
    return OptionSpec.from_flags("verbose", "--verbose", "-v", help="Show resolved LLM args on stderr.", default=False, is_flag=True, type=bool)


def llm_group() -> CommandGroupSpec:
    """Return the built-in LLM command group specification.

    Args:
        None.

    Returns:
        CommandGroupSpec: Backend-neutral CLI command metadata for chat,
            embeddings, image generation, interactive sessions, and LLM
            catalog inspection.
    """
    input_opt = OptionSpec.from_flags("input_files", "--input", "-i", help="Input file.", multiple=True)
    image_opt = OptionSpec.from_flags("image_files", "--image", help="Image path, URL, data URL, or base64 input.", multiple=True)
    json_opt = OptionSpec.from_flags("as_json", "--json", "-j", help="Print JSON.", default=False, is_flag=True, type=bool)
    copy_opt = OptionSpec.from_flags("copy", "--copy", "-cp", help="Copy output to the clipboard.", default=False, is_flag=True, type=bool)
    stream_opt = OptionSpec.from_flags("stream", "--stream", "--no-stream", help="Stream output.", default=True, is_flag=True, type=bool, negatable=True)
    return CommandGroupSpec(
        "llm",
        help="Call configured LLMs.",
        commands=[
            CommandSpec(
                "chat",
                _chat,
                help="Chat with an LLM.",
                args=[ArgumentSpec("prompt", help="Prompt.", required=False)],
                options=[
                    OptionSpec.from_flags("system", "--system", "-s", help="System prompt."),
                    input_opt,
                    image_opt,
                    stream_opt,
                    *_llm_options(),
                    _cache_option(),
                    OptionSpec.from_flags("mcps", "--mcp", help="MCP URL or registered Toolkit ref (namespace.toolkit:version).", multiple=True),
                    OptionSpec.from_flags("max_steps", "--max-steps", help="Maximum assistant steps for MCP tool loops.", default=_DEFAULT_MAX_STEPS, type=int),
                    json_opt,
                    copy_opt,
                    _verbose_option(),
                ],
            ),
            CommandSpec(
                "embed",
                _embed,
                help="Embed text.",
                args=[ArgumentSpec("text", help="Text.", required=False)],
                options=[
                    input_opt,
                    *_llm_options(),
                    _cache_option(),
                    OptionSpec.from_flags("preview", "--preview", help="Print an abbreviated embedding preview.", default=False, is_flag=True, type=bool),
                    json_opt,
                    copy_opt,
                    _verbose_option(),
                ],
            ),
            CommandSpec(
                "session",
                _session,
                help="Start an interactive chat session.",
                aliases=["sessions"],
                args=[ArgumentSpec("prompt", help="Initial prompt.", required=False)],
                options=[
                    OptionSpec.from_flags("system", "--system", "-s", help="System prompt."),
                    input_opt,
                    image_opt,
                    stream_opt,
                    *_llm_options(),
                    _cache_option(),
                    _verbose_option(),
                    OptionSpec.from_flags("mcps", "--mcp", help="MCP URL or registered Toolkit ref (namespace.toolkit:version).", multiple=True),
                ],
            ),
            CommandSpec(
                "imagen",
                _imagen,
                help="Generate an image.",
                aliases=["image"],
                args=[ArgumentSpec("prompt", help="Image prompt.")],
                options=[
                    OptionSpec.from_flags("output", "--output", "-o", help="Output image path."),
                    *_llm_options(),
                    _cache_option(),
                    json_opt,
                    _verbose_option(),
                ],
            ),
            CommandSpec("verbose", _verbose_cmd, help="Show resolved LLM args.", aliases=["resolve"], options=_llm_options()),
            CommandSpec("status", _status, help="Show selected LLM status.", options=_llm_options()),
            CommandSpec(
                "list",
                _list,
                help="List LLM presets/models/providers/gateways.",
                aliases=["ls"],
                args=[ArgumentSpec("catalog", help="Catalog name.", required=False, default="presets")],
            ),
        ],
    )
