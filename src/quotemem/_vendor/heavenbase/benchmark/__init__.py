"""HeavenBase benchmark package API."""

from __future__ import annotations

from heavenbase._lazy import _attach_lazy_exports

__all__ = [
    "BUCKET_BUILD",
    "BUCKET_EDIT",
    "BUCKET_QUERY",
    "BUCKET_REMOVAL",
    "BUCKETS",
    "MISSING_STAGE_ERROR",
    "AveragedResource",
    "AveragedRun",
    "AveragedStage",
    "BackendCombination",
    "BenchRun",
    "BenchmarkOp",
    "BenchmarkReport",
    "BenchmarkRunner",
    "BenchmarkScenario",
    "BucketSummary",
    "CombinationContext",
    "DefaultProductScenario",
    "ResourceSample",
    "ScenarioContext",
    "StageResult",
]

_LAZY_EXPORTS = {
    "BUCKET_BUILD": (".model", "BUCKET_BUILD"),
    "BUCKET_EDIT": (".model", "BUCKET_EDIT"),
    "BUCKET_QUERY": (".model", "BUCKET_QUERY"),
    "BUCKET_REMOVAL": (".model", "BUCKET_REMOVAL"),
    "BUCKETS": (".model", "BUCKETS"),
    "MISSING_STAGE_ERROR": (".model", "MISSING_STAGE_ERROR"),
    "AveragedResource": (".model", "AveragedResource"),
    "AveragedRun": (".model", "AveragedRun"),
    "AveragedStage": (".model", "AveragedStage"),
    "BenchRun": (".model", "BenchRun"),
    "BucketSummary": (".model", "BucketSummary"),
    "ResourceSample": (".model", "ResourceSample"),
    "StageResult": (".model", "StageResult"),
    "BenchmarkReport": (".report", "BenchmarkReport"),
    "BackendCombination": (".combination", "BackendCombination"),
    "BenchmarkRunner": (".runner", "BenchmarkRunner"),
    "CombinationContext": (".combination", "CombinationContext"),
    "BenchmarkOp": (".scenario", "BenchmarkOp"),
    "BenchmarkScenario": (".scenario", "BenchmarkScenario"),
    "DefaultProductScenario": (".scenario", "DefaultProductScenario"),
    "ScenarioContext": (".scenario", "ScenarioContext"),
}

__getattr__, __dir__ = _attach_lazy_exports(__name__, globals(), _LAZY_EXPORTS, __all__)
