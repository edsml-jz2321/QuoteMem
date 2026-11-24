"""Backend-neutral command specifications for package CLIs."""

from __future__ import annotations

__all__ = ["ArgumentSpec", "CommandGroupSpec", "CommandSpec", "OptionSpec"]

from heavenbase.utils import Any, Callable, Dict, List, Optional, Tuple, dataclass, field

from .base import CLIContext


@dataclass(frozen=True)
class ArgumentSpec:
    """One positional CLI argument."""

    name: str
    help: str = ""
    type: Any = str
    default: Any = None
    required: bool = True
    nargs: Optional[int | str] = None

    def is_variadic(self) -> bool:
        """Return whether this argument consumes zero or more values.

        Args:
            None.

        Returns:
            bool: ``True`` when ``nargs`` is ``-1`` or ``"*"``.
        """
        return self.nargs in (-1, "*")


@dataclass(frozen=True)
class OptionSpec:
    """One backend-neutral named CLI option.

    Attributes:
        name: Callback keyword receiving the parsed value.
        flags: Public flag spellings such as ``("--profile", "-p")``.
        help: Help text shared by every backend.
        type: Explicit value converter. When omitted, Click infers its
            converter from the default and the other backends parse text.
        default: Parsed value when the option is absent.
        is_flag: Whether the option is a value-free boolean switch.
        multiple: Whether the option may collect repeated values.
        negatable: Whether the option exposes positive and negative switches.
        hidden: Whether help output omits the option while parsing remains
            supported.
    """

    name: str
    flags: Tuple[str, ...]
    help: str = ""
    type: Any = None
    default: Any = None
    is_flag: bool = False
    multiple: bool = False
    negatable: bool = False
    hidden: bool = False

    @classmethod
    def from_flags(cls, name: str, *flags: str, **kwargs: Any) -> "OptionSpec":
        """Create an option from flag strings.

        Args:
            name (str): Internal option name used in parsed value dictionaries.
            *flags (str): CLI flag strings such as ``"--verbose"`` or
                ``"--open/--no-open"``.
            **kwargs (Any): Remaining ``OptionSpec`` fields. When the first
                flag contains ``"/"``, ``negatable`` defaults to ``True``.

        Returns:
            OptionSpec: Frozen option descriptor for backend compilers.
        """
        if "negatable" not in kwargs and flags and "/" in flags[0]:
            kwargs["negatable"] = True
        return cls(name=name, flags=tuple(flags), **kwargs)

    def flag_names(self) -> Tuple[str, ...]:
        """Return explicit flags or the conventional long flag.

        Args:
            None.

        Returns:
            Tuple[str, ...]: Explicit ``flags`` or a generated ``--name`` flag.
        """
        return self.flags or (f"--{self.name.replace('_', '-')}",)

    def positive_flags(self) -> Tuple[str, ...]:
        """Return flags that enable this option.

        Args:
            None.

        Returns:
            Tuple[str, ...]: All flags for normal options, or the positive
                half of a negatable ``flag/flag`` pair.
        """
        flags = self.flag_names()
        if not self.negatable:
            return flags
        return (flags[0].split("/", 1)[0],)

    def negative_flag(self) -> Optional[str]:
        """Return the flag that disables this option, when supported.

        Args:
            None.

        Returns:
            str | None: Negative flag for negatable options, otherwise ``None``.
        """
        if not self.negatable:
            return None
        flags = self.flag_names()
        if "/" in flags[0]:
            return flags[0].split("/", 1)[1] or None
        negative = next((flag for flag in flags[1:] if flag.startswith("--no-")), None)
        if negative:
            return negative
        positive = self.positive_flags()[0]
        return f"--no-{positive[2:]}" if positive.startswith("--") else None

    def click_flags(self) -> Tuple[str, ...]:
        """Return backend flags for Click-compatible boolean pairs.

        Args:
            None.

        Returns:
            Tuple[str, ...]: Explicit flags, or one combined ``positive/negative``
                string for negatable boolean options.
        """
        if not self.negatable:
            return self.flag_names()
        negative = self.negative_flag()
        positive = self.positive_flags()[0]
        return (f"{positive}/{negative}",) if negative else (positive,)


@dataclass(frozen=True)
class CommandSpec:
    """One backend-neutral command."""

    name: str
    callback: Callable[..., Any]
    help: str = ""
    args: List[ArgumentSpec] = field(default_factory=list)
    options: List[OptionSpec] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def invoke(self, context: CLIContext, values: Dict[str, Any]) -> Any:
        """Invoke the command callback with context and parsed values.

        Args:
            context (CLIContext): Runtime context passed to the callback.
            values (Dict[str, Any]): Parsed positional and option values keyed
                by ``ArgumentSpec.name`` and ``OptionSpec.name``.

        Returns:
            Any: Return value from the command callback, if any.

        Raises:
            SystemExit: Re-raised when the callback exits the process.
        """
        try:
            return self.callback(context=context, **values)
        except SystemExit:
            raise
        except KeyboardInterrupt as error:
            context.fail(error, code=130, msg="interrupted")
        except Exception as error:
            context.fail(error)


@dataclass
class CommandGroupSpec:
    """A command group such as config, ws, or llm."""

    name: str
    help: str = ""
    commands: List[CommandSpec] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def command_names(self) -> List[str]:
        """Return canonical command names in this group.

        Args:
            None.

        Returns:
            List[str]: Canonical ``CommandSpec.name`` values in declaration order.
        """
        return [command.name for command in self.commands]
