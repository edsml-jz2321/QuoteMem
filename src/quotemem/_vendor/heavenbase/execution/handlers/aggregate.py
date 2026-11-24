"""Shared aggregate compiler descriptors and typed route declines."""

from __future__ import annotations

__all__ = ["AggregateCompileResult", "AggregateCompiler", "AggregateCompilerFn", "AggregateCompilerInput", "AggregateMetricInput"]

from dataclasses import InitVar

from heavenbase.utils import Any, Callable, Mapping, Optional, Protocol, Sequence, dataclass

from ...filters.ops import normalize_op
from ...modules import ModuleResolver, module_resolver
from ..contracts import HandlerContext, QueryFragment
from .base import backend_identifier


class AggregateMetricInput(Protocol):
    """Structural reducer target accepted by handler-layer aggregate compilers."""

    field: str
    op: str
    options: Mapping[str, Any]


class AggregateCompilerInput(Protocol):
    """Structural aggregate request accepted without importing the query layer."""

    metrics: Mapping[str, AggregateMetricInput]
    group_by: Sequence[str]


AggregateCompilerFn = Callable[[Any, AggregateCompilerInput, Optional[QueryFragment], HandlerContext], "AggregateCompileResult"]


@dataclass(frozen=True)
class AggregateCompileResult:
    """Return either one provider-ready fragment or a named pre-execution decline.

    Args:
        fragment: Provider-ready aggregate fragment for native execution.
        reason: Stable fold reason when compilation is declined.
    """

    fragment: Optional[QueryFragment] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Require exactly one native fragment or decline reason."""
        if (self.fragment is None) == (self.reason is None):
            raise ValueError("AggregateCompileResult needs exactly one fragment or reason")
        if self.fragment is not None and not isinstance(self.fragment, QueryFragment):
            raise TypeError("AggregateCompileResult fragment must be a QueryFragment")
        if self.reason is not None:
            reason = str(self.reason).strip()
            if not reason:
                raise ValueError("Aggregate compile decline reason must not be empty")
            object.__setattr__(self, "reason", reason)

    @classmethod
    def native(cls, fragment: QueryFragment) -> "AggregateCompileResult":
        """Build a successful native compilation result.

        Args:
            fragment (QueryFragment): Provider-ready aggregate fragment.

        Returns:
            AggregateCompileResult: Native compilation result wrapping ``fragment``.
        """
        return cls(fragment=fragment)

    @classmethod
    def decline(cls, reason: str) -> "AggregateCompileResult":
        """Build a named pre-execution fold decision.

        Args:
            reason (str): Stable decline reason recorded before execution.

        Returns:
            AggregateCompileResult: Declined compilation result carrying ``reason``.
        """
        return cls(reason=reason)


@dataclass(frozen=True)
class AggregateCompiler:
    """Compile a complete validated aggregate for one backend type.

    Args:
        backend_identifier: Registered backend type.
        operations: Reducer operations this compiler can lower.
        compile: Pure handler callable returning a fragment or typed decline.
        resolver: Context-owned resolver used to normalize capability identity.
            It is not retained by the descriptor.
    """

    backend_identifier: str
    operations: Sequence[str]
    compile: AggregateCompilerFn
    resolver: InitVar[ModuleResolver | None] = None

    def __post_init__(self, resolver: ModuleResolver | None) -> None:
        """Normalize capability identity and validate the compiler callable."""
        selected = module_resolver(resolver)
        object.__setattr__(self, "backend_identifier", backend_identifier(self.backend_identifier, resolver=selected))
        operations = tuple(dict.fromkeys(normalize_op(op, resolver=selected) for op in self.operations))
        if not operations:
            raise ValueError("AggregateCompiler needs at least one reducer operation")
        object.__setattr__(self, "operations", operations)
        if not callable(self.compile):
            raise TypeError("AggregateCompiler compile must be callable")

    def __call__(self, schema: Any, aggregate: AggregateCompilerInput, where: Optional[QueryFragment], ctx: HandlerContext) -> AggregateCompileResult:
        """Compile when every requested reducer is registered for this provider."""
        unsupported = sorted({metric.op for metric in aggregate.metrics.values()} - set(self.operations))
        if unsupported:
            return AggregateCompileResult.decline(f"backend_reducer_not_registered:{','.join(unsupported)}")
        result = self.compile(schema, aggregate, where, ctx)
        if not isinstance(result, AggregateCompileResult):
            raise TypeError("Aggregate compiler must return AggregateCompileResult")
        if result.fragment is not None and result.fragment.backend != ctx.backend:
            raise ValueError(f"Aggregate compiler returned backend {result.fragment.backend!r}; expected {ctx.backend!r}")
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return registry-derived native capability metadata.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable backend identifier, operations, and strategy.
        """
        return {"backend_identifier": self.backend_identifier, "operations": list(self.operations), "strategy": "inline-column"}
