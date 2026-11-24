"""Benchmark data model shared by runners and reporters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

BUCKET_BUILD = "build"
BUCKET_QUERY = "query"
BUCKET_EDIT = "edit"
BUCKET_REMOVAL = "removal"
BUCKETS = (BUCKET_BUILD, BUCKET_QUERY, BUCKET_EDIT, BUCKET_REMOVAL)

MISSING_STAGE_ERROR = "MissingStage"
MISSING_STAGE_MESSAGE = "repeat aborted before this stage ran"


@dataclass
class StageResult:
    """One timed benchmark operation."""

    name: str
    bucket: str
    elapsed_sec: float
    rows: int = 0
    notes: str = ""
    error_type: str = ""
    error_message: str = ""

    @property
    def throughput(self) -> float | None:
        """Return rows processed per second when row timing is available."""
        if self.rows > 0 and self.elapsed_sec > 0:
            return self.rows / self.elapsed_sec
        return None

    @property
    def latency_ms(self) -> float | None:
        """Return average per-row latency in milliseconds when available."""
        if self.rows > 0 and self.elapsed_sec > 0:
            return (self.elapsed_sec / self.rows) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this stage result.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable stage metrics and error metadata.
        """
        return {
            "name": self.name,
            "bucket": self.bucket,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "rows": self.rows,
            "throughput": round(self.throughput, 2) if self.throughput is not None else None,
            "latency_ms": round(self.latency_ms, 4) if self.latency_ms is not None else None,
            "notes": self.notes,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass
class ResourceSample:
    """Best-effort memory and disk sample around a benchmark phase."""

    name: str
    memory_mb: float | None = None
    disk_mb: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this resource sample.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable memory and disk sample fields.
        """
        return {"name": self.name, "memory_mb": self.memory_mb, "disk_mb": round(self.disk_mb, 3), "notes": self.notes}


@dataclass
class BenchRun:
    """Raw result for one combination/scale/repeat run."""

    combination_label: str
    combination_type: str
    backend_identifiers: dict[str, str]
    scenario_label: str
    scale: int
    repeat: int
    total_rows: int
    stages: list[StageResult] = field(default_factory=list)
    resources: list[ResourceSample] = field(default_factory=list)

    @property
    def total_sec(self) -> float:
        """Return total elapsed seconds across all stages."""
        return sum(stage.elapsed_sec for stage in self.stages)

    def bucket_sec(self, bucket: str) -> float:
        """Return elapsed seconds for stages in one benchmark bucket.

        Args:
            bucket (str): Benchmark bucket name such as ``build`` or ``query``.

        Returns:
            float: Sum of stage elapsed seconds in the bucket.
        """
        return sum(stage.elapsed_sec for stage in self.stages if stage.bucket == bucket)

    def errors(self) -> list[StageResult]:
        """Return stages that captured execution errors.

        Args:
            None.

        Returns:
            list[StageResult]: Stages whose ``error_type`` is set.
        """
        return [stage for stage in self.stages if stage.error_type]

    def resource(self, name: str) -> ResourceSample | None:
        """Return the named resource sample when the run captured it.

        Args:
            name (str): Resource sample name such as ``before_build``.

        Returns:
            ResourceSample | None: Matching sample, or ``None`` when absent.
        """
        return next((sample for sample in self.resources if sample.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this raw run.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable run metadata, stages, and resources.
        """
        return {
            "combination_label": self.combination_label,
            "combination_type": self.combination_type,
            "backend_identifiers": dict(self.backend_identifiers),
            "scenario_label": self.scenario_label,
            "scale": self.scale,
            "repeat": self.repeat,
            "total_rows": self.total_rows,
            "total_sec": round(self.total_sec, 6),
            "bucket_sec": {bucket: round(self.bucket_sec(bucket), 6) for bucket in BUCKETS},
            "resources": [sample.to_dict() for sample in self.resources],
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass
class AveragedStage:
    """Stage metrics averaged across the repeats that measured the stage.

    The mean describes `samples` of the group's `repeats`, never all of them by
    assumption: a repeat that raised, and a repeat that aborted earlier and so
    never reached this stage, both contribute no timing. Readers must be able to
    see that gap without inferring it, so the sample count travels with the mean
    and :attr:`partial` states the verdict.

    `mean_sec` is `None` when no repeat measured the stage. A substituted `0.0`
    would publish an instantaneous stage and an infinite throughput for work that
    never ran.
    """

    name: str
    bucket: str
    mean_sec: float | None
    std_sec: float | None
    rows: int
    samples: int
    repeats: int
    notes: str
    errors: list[dict[str, Any]] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        """Return whether the mean covers fewer repeats than the group ran.

        Returns:
            bool: `True` when at least one repeat contributed no timing.
        """
        return self.samples < self.repeats

    @property
    def throughput(self) -> float | None:
        """Return mean rows processed per second when row timing is available."""
        if self.rows > 0 and self.mean_sec is not None and self.mean_sec > 0:
            return self.rows / self.mean_sec
        return None

    @property
    def latency_ms(self) -> float | None:
        """Return mean per-row latency in milliseconds when available."""
        if self.rows > 0 and self.mean_sec is not None and self.mean_sec > 0:
            return (self.mean_sec / self.rows) * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this averaged stage.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable averaged stage metrics, coverage, and errors.
        """
        return {
            "name": self.name,
            "bucket": self.bucket,
            "mean_sec": None if self.mean_sec is None else round(self.mean_sec, 6),
            "std_sec": None if self.std_sec is None else round(self.std_sec, 6),
            "rows": self.rows,
            "samples": self.samples,
            "repeats": self.repeats,
            "partial": self.partial,
            "throughput": round(self.throughput, 2) if self.throughput is not None else None,
            "latency_ms": round(self.latency_ms, 4) if self.latency_ms is not None else None,
            "notes": self.notes,
            "errors": list(self.errors),
        }


@dataclass
class AveragedResource:
    """Resource sample averaged across the repeats that reached the checkpoint.

    A repeat that aborted before a checkpoint records no sample there, so two
    checkpoints of the same aggregate can rest on different populations. Each
    averaged field therefore carries its own sample count: `samples` backs
    `disk_mb`, and `memory_samples` backs `memory_mb`, which is additionally
    absent whenever process RSS could not be read.
    """

    name: str
    memory_mb: float | None
    disk_mb: float | None
    samples: int
    memory_samples: int
    repeats: int
    notes: str = ""

    @property
    def partial(self) -> bool:
        """Return whether fewer repeats reached this checkpoint than the group ran.

        Returns:
            bool: `True` when at least one repeat contributed no sample.
        """
        return self.samples < self.repeats

    def samples_for(self, field_name: str) -> int:
        """Return how many raw samples backed one averaged field.

        Args:
            field_name (str): Averaged field name, such as `memory_mb` or `disk_mb`.

        Returns:
            int: Number of raw samples the named mean was taken over.
        """
        return self.memory_samples if field_name == "memory_mb" else self.samples

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this averaged sample.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable averaged resource fields and their sample counts.
        """
        return {
            "name": self.name,
            "memory_mb": self.memory_mb,
            "disk_mb": None if self.disk_mb is None else round(self.disk_mb, 3),
            "samples": self.samples,
            "memory_samples": self.memory_samples,
            "repeats": self.repeats,
            "partial": self.partial,
            "notes": self.notes,
        }


@dataclass
class BucketSummary:
    """Aggregated timing for one objective bucket.

    The total sums `measured` of the bucket's `stages`. A stage no repeat ever
    timed contributes nothing, which silently shrinks the bucket, so the stage
    count travels with the total and :attr:`partial` states the verdict.
    """

    bucket: str
    mean_sec: float | None
    std_sec: float | None
    stages: int
    measured: int

    @property
    def partial(self) -> bool:
        """Return whether some stage in this bucket contributed no timing.

        Returns:
            bool: `True` when the total omits at least one of the bucket's stages.
        """
        return self.measured < self.stages

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this bucket summary.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable bucket timing aggregate and its stage coverage.
        """
        return {
            "bucket": self.bucket,
            "mean_sec": None if self.mean_sec is None else round(self.mean_sec, 6),
            "std_sec": None if self.std_sec is None else round(self.std_sec, 6),
            "stages": self.stages,
            "measured": self.measured,
            "partial": self.partial,
        }


@dataclass
class AveragedRun:
    """A combination/scale aggregate across repeats.

    `repeats` counts the runs that were launched, which is not the same as the
    runs that reached any given stage or checkpoint. Per-stage and per-resource
    coverage lives on the aggregates themselves; :attr:`samples` against
    :attr:`expected_samples` summarises it for the whole aggregate.
    """

    combination_label: str
    combination_type: str
    backend_identifiers: dict[str, str]
    scenario_label: str
    scale: int
    repeats: int
    total_rows: int
    stages: list[AveragedStage]
    resources: list[AveragedResource]

    @property
    def measured_stages(self) -> list[AveragedStage]:
        """Return the averaged stages that at least one repeat actually timed."""
        return [stage for stage in self.stages if stage.mean_sec is not None]

    @property
    def samples(self) -> int:
        """Return how many stage repeats produced a timing in this aggregate."""
        return sum(stage.samples for stage in self.stages)

    @property
    def expected_samples(self) -> int:
        """Return how many stage repeats this aggregate set out to measure."""
        return sum(stage.repeats for stage in self.stages)

    @property
    def total_sec(self) -> float:
        """Return total mean elapsed seconds across the stages that were measured."""
        return sum(stage.mean_sec for stage in self.measured_stages)

    @property
    def total_std(self) -> float:
        """Return combined standard deviation across the stages that were measured."""
        return math.sqrt(sum(stage.std_sec**2 for stage in self.measured_stages))

    def bucket(self, name: str) -> BucketSummary:
        """Return aggregate timing for one benchmark bucket.

        Stages no repeat timed are excluded rather than summed as zero, and the
        returned summary states how many of the bucket's stages the total covers.

        Args:
            name (str): Benchmark bucket name such as ``build`` or ``query``.

        Returns:
            BucketSummary: Total and standard deviation over the measured stages in the
            bucket, with `mean_sec` and `std_sec` set to `None` when none were measured.
        """
        stages = [stage for stage in self.stages if stage.bucket == name]
        measured = [stage for stage in stages if stage.mean_sec is not None]
        mean = sum(stage.mean_sec for stage in measured) if measured else None
        std = math.sqrt(sum(stage.std_sec**2 for stage in measured)) if measured else None
        return BucketSummary(name, mean, std, len(stages), len(measured))

    def resource(self, name: str) -> AveragedResource | None:
        """Return the named averaged resource sample when present.

        Args:
            name (str): Resource sample name such as ``after_build``.

        Returns:
            AveragedResource | None: Matching averaged sample, or ``None`` when absent.
        """
        return next((sample for sample in self.resources if sample.name == name), None)

    def resource_delta(self, before: str, after: str, field_name: str) -> float | None:
        """Return a resource metric delta between two named samples.

        A repeat that aborted mid-run reaches the earlier checkpoint and not the
        later one, so the two means can rest on different populations. Subtracting
        across them measures the population change, not the phase, so this returns
        `None` instead: an absent delta is honest, a mixed one is not.

        Args:
            before (str): Earlier resource sample name.
            after (str): Later resource sample name.
            field_name (str): Resource field to compare, such as ``memory_mb`` or ``disk_mb``.

        Returns:
            float | None: ``after - before`` when both means exist and cover the same
            number of repeats, else ``None``.
        """
        left = self.resource(before)
        right = self.resource(after)
        if left is None or right is None:
            return None
        lval = getattr(left, field_name)
        rval = getattr(right, field_name)
        if lval is None or rval is None:
            return None
        if left.samples_for(field_name) != right.samples_for(field_name):
            return None
        return rval - lval

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation of this averaged run.

        Args:
            None.

        Returns:
            dict[str, Any]: Serializable averaged run metadata, buckets, and stages.
        """
        return {
            "combination_label": self.combination_label,
            "combination_type": self.combination_type,
            "backend_identifiers": dict(self.backend_identifiers),
            "scenario_label": self.scenario_label,
            "scale": self.scale,
            "repeats": self.repeats,
            "total_rows": self.total_rows,
            "samples": self.samples,
            "expected_samples": self.expected_samples,
            "total_sec": round(self.total_sec, 6),
            "total_std": round(self.total_std, 6),
            "buckets": {bucket: self.bucket(bucket).to_dict() for bucket in BUCKETS},
            "resources": [sample.to_dict() for sample in self.resources],
            "resource_deltas": {
                "build_memory_mb": self.resource_delta("before_build", "after_build", "memory_mb"),
                "build_disk_mb": self.resource_delta("before_build", "after_build", "disk_mb"),
                "edit_memory_mb": self.resource_delta("after_build", "after_edit", "memory_mb"),
                "edit_disk_mb": self.resource_delta("after_build", "after_edit", "disk_mb"),
                "removal_memory_mb": self.resource_delta("after_edit", "after_removal", "memory_mb"),
                "removal_disk_mb": self.resource_delta("after_edit", "after_removal", "disk_mb"),
            },
            "stages": [stage.to_dict() for stage in self.stages],
        }
