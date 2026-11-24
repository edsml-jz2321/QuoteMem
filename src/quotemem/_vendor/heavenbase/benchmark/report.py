"""JSON and Markdown benchmark report composition."""

from __future__ import annotations

__all__ = [
    "BenchmarkReport",
    "_average_stage",
]

import statistics
import time
from typing import Any

from heavenbase.utils.files import touch_file
from heavenbase.utils.serialize import dump_json

from .model import (
    BUCKETS,
    MISSING_STAGE_ERROR,
    MISSING_STAGE_MESSAGE,
    AveragedResource,
    AveragedRun,
    AveragedStage,
    BenchRun,
    BucketSummary,
    ResourceSample,
    StageResult,
)


def _fmt(val: float | None, unit: str = "") -> str:
    if val is None:
        return "-"
    if val >= 1000:
        return f"{val:,.0f}{unit}"
    if val >= 10:
        return f"{val:,.1f}{unit}"
    return f"{val:,.3f}{unit}"


def _time_str(mean: float | None, std: float | None = 0.0) -> str:
    if mean is None:
        return "-"
    if std:
        return f"{mean:.3f} ± {std:.3f}"
    return f"{mean:.3f}"


def _bucket_str(summary: BucketSummary) -> str:
    text = _time_str(summary.mean_sec, summary.std_sec)
    if summary.partial:
        return f"{text} ({summary.measured}/{summary.stages} stages)"
    return text


def _average_stage(name: str, repeats: list[StageResult]) -> AveragedStage:
    successful = [stage for stage in repeats if not stage.error_type]
    times = [stage.elapsed_sec for stage in successful]
    mean = statistics.mean(times) if times else None
    std = (statistics.stdev(times) if len(times) > 1 else 0.0) if times else None
    first = successful[0] if successful else repeats[0]
    errors = [{"repeat": pos, "type": stage.error_type, "message": stage.error_message} for pos, stage in enumerate(repeats, 1) if stage.error_type]
    notes = first.notes
    if errors:
        shown = "; ".join(f"r{err['repeat']} {err['type']}: {err['message']}" for err in errors[:3])
        suffix = "" if len(errors) <= 3 else f"; +{len(errors) - 3} more"
        notes = (
            f"partial mean over {len(successful)}/{len(repeats)} successful repeats; errors: {shown}{suffix}"
            if successful
            else f"ERROR repeats: {shown}{suffix}"
        )
    return AveragedStage(name, first.bucket, mean, std, first.rows, len(successful), len(repeats), notes, errors)


def _average_resource(name: str, samples: list[ResourceSample], repeats: int) -> AveragedResource:
    mem_values = [sample.memory_mb for sample in samples if sample.memory_mb is not None]
    memory_mb = statistics.mean(mem_values) if mem_values else None
    disk_mb = statistics.mean([sample.disk_mb for sample in samples]) if samples else None
    notes = "; ".join(sorted({sample.notes for sample in samples if sample.notes}))
    if len(samples) < repeats:
        notes = "; ".join(part for part in (f"partial mean over {len(samples)}/{repeats} repeats", notes) if part)
    return AveragedResource(name, memory_mb, disk_mb, len(samples), len(mem_values), repeats, notes)


def _missing_stage(template: StageResult) -> StageResult:
    """Return the placeholder for a repeat that never reached one stage.

    The placeholder borrows the stage identity from a repeat that did record it,
    so an aborted repeat can never redefine the stage's bucket or row count, and
    it carries an error so the averaging path excludes it rather than averaging in
    a zero nothing measured.

    Args:
        template (StageResult): Any repeat's real result for the same stage.

    Returns:
        StageResult: Errored placeholder standing in for the absent measurement.
    """
    return StageResult(
        template.name,
        template.bucket,
        0.0,
        template.rows,
        notes=MISSING_STAGE_MESSAGE,
        error_type=MISSING_STAGE_ERROR,
        error_message=MISSING_STAGE_MESSAGE,
    )


class BenchmarkReport:
    """Structured report returned by benchmark runs."""

    def __init__(self, runs: list[BenchRun] | None = None, meta: dict[str, Any] | None = None):
        """Create a benchmark report from optional raw runs and metadata.

        Args:
            runs (list[BenchRun] | None): Initial raw benchmark runs. Defaults to ``None``.
            meta (dict[str, Any] | None): Report metadata such as run id and scenario label. Defaults to ``None``.

        Returns:
            None: This function does not return a value.
        """
        self.runs = list(runs or [])
        self.meta = dict(meta or {})

    def add(self, run: BenchRun) -> None:
        """Append one raw benchmark run to the report.

        Args:
            run (BenchRun): Captured benchmark run to append.

        Returns:
            None: This method does not return a value.
        """
        self.runs.append(run)

    def has_errors(self) -> bool:
        """Return whether any captured stage failed during benchmarking.

        Args:
            None.

        Returns:
            bool: ``True`` when at least one stage recorded an error.
        """
        return any(stage.error_type for run in self.runs for stage in run.stages)

    def error_count(self) -> int:
        """Return the number of captured stage failures.

        Args:
            None.

        Returns:
            int: Count of stages whose ``error_type`` is set.
        """
        return sum(1 for run in self.runs for stage in run.stages if stage.error_type)

    def aggregated(self) -> list[AveragedRun]:
        """Return run results averaged by backend combination and scale.

        Args:
            None.

        Returns:
            list[AveragedRun]: Aggregates grouped by combination label and scale.
        """
        grouped: dict[tuple[str, int], list[BenchRun]] = {}
        for run in self.runs:
            grouped.setdefault((run.combination_label, run.scale), []).append(run)

        out: list[AveragedRun] = []
        for _, runs in sorted(grouped.items()):
            stage_names = []
            for run in runs:
                for stage in run.stages:
                    if stage.name not in stage_names:
                        stage_names.append(stage.name)
            stages = []
            for name in stage_names:
                recorded = [stage for run in runs for stage in run.stages if stage.name == name]
                placeholder = _missing_stage(recorded[0])
                repeats = [next((stage for stage in run.stages if stage.name == name), placeholder) for run in runs]
                stages.append(_average_stage(name, repeats))

            resource_names = []
            for run in runs:
                for sample in run.resources:
                    if sample.name not in resource_names:
                        resource_names.append(sample.name)
            resources = []
            for name in resource_names:
                samples = [sample for run in runs for sample in run.resources if sample.name == name]
                resources.append(_average_resource(name, samples, len(runs)))

            first = runs[0]
            out.append(
                AveragedRun(
                    first.combination_label,
                    first.combination_type,
                    dict(first.backend_identifiers),
                    first.scenario_label,
                    first.scale,
                    len(runs),
                    first.total_rows,
                    stages,
                    resources,
                )
            )
        return out

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        """Return a JSON-safe benchmark report payload.

        Args:
            include_raw (bool): Whether to include every raw run. Defaults to ``False``.

        Returns:
            dict[str, Any]: Aggregated combinations plus report metadata.
        """
        payload = {
            "meta": {**self.meta, "stage_errors": self.error_count()},
            "combinations": [run.to_dict() for run in self.aggregated()],
        }
        if include_raw:
            payload["raw_runs"] = [run.to_dict() for run in self.runs]
        return payload

    def write_json(self, path: str, *, include_raw: bool = False) -> None:
        """Write the benchmark report payload to a JSON file.

        Args:
            path (str): Destination JSON file path.
            include_raw (bool): Whether to include every raw run. Defaults to ``False``.

        Returns:
            None: This method does not return a value.
        """
        dump_json(self.to_dict(include_raw=include_raw), path)

    def to_markdown(self) -> str:
        """Render the benchmark report as Markdown.

        Args:
            None.

        Returns:
            str: Markdown report text.
        """
        agg = self.aggregated()
        generated = self.meta.get("generated_at") or time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# HeavenBase Benchmark Report",
            "",
            f"- **Generated**: {generated}",
            f"- **Run ID**: {self.meta.get('run_id', '-')}",
            f"- **Scenario**: {self.meta.get('scenario', '-')}",
            f"- **System**: {', '.join(f'{k}={v}' for k, v in (self.meta.get('sys_info') or {}).items())}",
            f"- **Scales**: {', '.join(str(scale) + 'x' for scale in self.meta.get('scales', []))}",
            f"- **Repeats per scale**: {self.meta.get('repeats', '-')}",
            f"- **Seed**: {self.meta.get('seed', '-')}",
            f"- **Wall time**: {self.meta.get('wall_sec', 0.0):.1f}s",
            f"- **Stage errors**: {self.error_count()}",
        ]
        if self.error_count():
            lines.append("")
            lines.append(
                f"> **WARNING**: {self.error_count()} stage error(s) in this run. Every mean below covers only the repeats that "
                "reached and completed its stage; read the `Samples` columns before comparing this report with any other."
            )
        lines.extend(
            [
                "",
                "## Summary",
                "",
                "| Combination | Type | Scale | Rows | Repeats | Samples | Build (s) | Query (s) | Edit (s) | Removal (s) | Mem Build Delta (MB) | Disk Build Delta (MB) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in agg:
            mem_delta = run.resource_delta("before_build", "after_build", "memory_mb")
            disk_delta = run.resource_delta("before_build", "after_build", "disk_mb")
            buckets = " | ".join(_bucket_str(run.bucket(name)) for name in BUCKETS)
            lines.append(
                f"| {run.combination_label} | {run.combination_type} | {run.scale}x | {run.total_rows:,} | {run.repeats} | "
                f"{run.samples}/{run.expected_samples} | {buckets} | {_fmt(mem_delta)} | {_fmt(disk_delta)} |"
            )
        lines.extend(["", "---", ""])

        for run in agg:
            lines.extend(
                [
                    f"## {run.combination_label} (`{run.combination_type}`) - scale={run.scale}x",
                    "",
                    f"- **Rows**: {run.total_rows:,}",
                    f"- **Repeats**: {run.repeats}",
                    f"- **Stage samples**: {run.samples}/{run.expected_samples} measured",
                    f"- **Total time**: {_time_str(run.total_sec, run.total_std)}s over {len(run.measured_stages)}/{len(run.stages)} stages",
                    f"- **Backends**: {', '.join(f'{name}={kind}' for name, kind in sorted(run.backend_identifiers.items()))}",
                    "",
                    "### Resources",
                    "",
                    "| Sample | Memory (MB) | Disk (MB) | Samples | Notes |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            for sample in run.resources:
                coverage = f"{sample.samples}/{sample.repeats}"
                lines.append(f"| {sample.name} | {_fmt(sample.memory_mb)} | {_fmt(sample.disk_mb)} | {coverage} | {sample.notes} |")
            lines.extend(
                [
                    "",
                    "### Stages",
                    "",
                    "| Bucket | Stage | Mean (s) | Std (s) | Samples | Rows | Throughput | Latency (ms) | Notes |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for stage in run.stages:
                rows_s = f"{stage.rows:,}" if stage.rows else "-"
                lines.append(
                    f"| {stage.bucket} | {stage.name} | {_time_str(stage.mean_sec)} | {_time_str(stage.std_sec)} | "
                    f"{stage.samples}/{stage.repeats} | {rows_s} | "
                    f"{_fmt(stage.throughput, ' r/s')} | {_fmt(stage.latency_ms, ' ms')} | {stage.notes} |"
                )
            lines.extend(["", "---", ""])

        lines.extend(
            [
                "## How to read this report",
                "",
                "- **Build**: workspace/backend setup plus initial bulk upsert.",
                "- **Query**: read, lookup, scalar filter, boolean filter, vector near, and near+filter operations.",
                "- **Edit**: data patch/update operations such as `set_many`.",
                "- **Removal**: deletion and post-delete validation stages.",
                "- **Resource samples**: process RSS and backend path size sampled before build, after build, after edit, and after removal.",
                "- **Stage errors**: failed repeats are excluded from stage means and listed in the notes.",
                "- **Samples**: repeats that produced a measurement, out of the repeats the group ran. A repeat that failed, or that "
                "aborted before reaching a stage, is excluded rather than counted as zero, so any `Samples` below its `Repeats` marks "
                "a mean taken over a subset. `-` means nothing was measured at all.",
                "- **Deltas**: a resource delta is published only when both checkpoints were averaged over the same repeats; otherwise "
                "it prints `-`, because subtracting means taken over different populations measures the population, not the phase.",
                "- **Bucket totals**: `(m/n stages)` beside a Build/Query/Edit/Removal total means the total omits stages no repeat measured.",
                "",
                "## Running the benchmark",
                "",
                "```bash",
                "bash scripts/benchmark.bash",
                "bash scripts/benchmark.bash --scales 1 --repeats 1",
                "bash scripts/benchmark.bash --combinations inmem_all,sqlite_all",
                "bash scripts/benchmark.bash --artifact result.json --report REPORT.md",
                "bash scripts/benchmark.bash --allow-errors",
                "```",
                "",
            ]
        )
        return "\n".join(lines)

    def write_markdown(self, path: str) -> None:
        """Write the benchmark report as Markdown.

        Args:
            path (str): Destination Markdown file path.

        Returns:
            None: This method does not return a value.
        """
        touch_file(path, clear=True, content=self.to_markdown())

    def print(self) -> None:
        """Print the Markdown benchmark report to stdout.

        Args:
            None.

        Returns:
            None: This method does not return a value.
        """
        print(self.to_markdown())
