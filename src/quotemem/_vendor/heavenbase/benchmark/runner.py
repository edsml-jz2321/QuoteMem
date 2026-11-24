"""Context-bound benchmark recipe resolution and execution."""

from __future__ import annotations

__all__ = ["BenchmarkRunner"]

from collections.abc import Mapping, Sequence
import gc
from genericpath import getsize
import platform
import time
from typing import Any
from uuid import uuid4

from heavenbase.utils import safe_error
from heavenbase.utils.files import delete_dir, enum_files, exists_dir, exists_file, pj, touch_dir
from heavenbase.utils.rng import StableRNG

from ..context import Context
from .combination import BackendCombination, CombinationContext
from .model import BUCKET_BUILD, BenchRun, ResourceSample, StageResult
from .report import BenchmarkReport
from .scenario import BenchmarkOp, BenchmarkScenario, DefaultProductScenario, ScenarioContext


def _parse_scales(value: str | Sequence[int]) -> list[int]:
    """Return validated positive scale factors."""
    scales = [int(item.strip()) for item in value.split(",") if item.strip()] if isinstance(value, str) else [int(item) for item in value]
    if not scales or any(scale < 1 for scale in scales):
        raise ValueError("benchmark scales must contain positive integers")
    return scales


class _Timer:
    """Small context manager for stage timing."""

    def __init__(self, name: str, bucket: str, rows: int = 0):
        self.name = name
        self.bucket = bucket
        self.rows = rows
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "_Timer":
        gc.collect()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.elapsed = time.perf_counter() - self._start
        return False

    def result(self, notes: str = "") -> StageResult:
        """Return the timed stage result for this timer."""
        return StageResult(self.name, self.bucket, self.elapsed, self.rows, notes)


def _stage(op: BenchmarkOp) -> StageResult:
    timer = _Timer(op.name, op.bucket, op.rows)
    try:
        with timer:
            op.fn()
        return timer.result(op.notes)
    except Exception as exc:
        msg = safe_error(exc)
        if len(msg) > 256:
            msg = msg[:253] + "..."
        return StageResult(op.name, op.bucket, timer.elapsed, op.rows, f"ERROR: {msg}", type(exc).__name__, msg)


def _rss_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _path_size_mb(path: str) -> float:
    if exists_file(path):
        return getsize(path) / (1024 * 1024)
    if exists_dir(path):
        return sum(getsize(file_path) for file_path in enum_files(path, abs=True)) / (1024 * 1024)
    return 0.0


def _resources(name: str, ctx: CombinationContext, combo: BackendCombination) -> ResourceSample:
    notes = []
    disk_mb = 0.0
    for path in combo.paths(ctx):
        try:
            disk_mb += _path_size_mb(path)
        except Exception as exc:
            notes.append(f"disk {path}: {safe_error(exc)}")
    mem_mb = _rss_mb()
    if mem_mb is None:
        notes.append("process RSS unavailable")
    return ResourceSample(name, mem_mb, disk_mb, "; ".join(notes))


def _sys_info() -> dict[str, str]:
    info = {"platform": platform.system(), "machine": platform.machine(), "processor": platform.processor() or "unknown", "python": platform.python_version()}
    try:
        import psutil

        info["ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"
    except Exception:
        info["ram_gb"] = "unknown"
    return info


def _sample(ids: Sequence[str], rng: StableRNG, k: int) -> list[str]:
    if not ids or k <= 0:
        return []
    with rng as gen:
        return list(gen.choice(list(ids), k=min(k, len(ids)), replace=False))


class BenchmarkRunner:
    """Run one workload through a caller-owned Context.

    Args:
        context (Context): Explicit composition authority for configuration,
            module recipes, Backend construction, and temporary workspaces.
        scenario (BenchmarkScenario | None): Workload definition. Defaults to
            the bundled product scenario configured by `context`.
        combinations (str | Sequence[str | BackendCombination] | None): Recipe
            identifiers, comma-separated identifiers, or direct recipes.
            Omitting this selects the registered `default` suite.
        scales (str | Sequence[int] | None): Positive workload multipliers.
        repeats (int | None): Repetitions per combination and scale. Must be at
            least one.
        seed (int | None): Stable random seed.
        temp_root (str | None): Root for run-owned temporary storage.
    """

    def __init__(
        self,
        context: Context,
        *,
        scenario: BenchmarkScenario | None = None,
        combinations: str | Sequence[str | BackendCombination] | None = None,
        scales: str | Sequence[int] | None = None,
        repeats: int | None = None,
        seed: int | None = None,
        temp_root: str | None = None,
    ) -> None:
        """Initialize this runner from one Context and optional overrides.

        Returns:
            None: This function does not return a value.
        """
        if not isinstance(context, Context):
            raise TypeError("BenchmarkRunner requires an explicit Context")
        self.context = context
        self.resolver = context.modules()
        self._settings = context.config.snapshot().get("heavenbase.benchmark", default={})
        self.scenario = scenario or DefaultProductScenario(
            base_rows=int(self._setting("base_rows", 1_000)),
            vector_dim=int(self._setting("vector_dim", 1024)),
            top_k=int(self._setting("top_k", 10)),
        )
        self.combinations = self._combinations(combinations)
        self.scales = _parse_scales(self._setting("scales", (1, 10)) if scales is None else scales)
        self.repeats = int(self._setting("repeats", 5) if repeats is None else repeats)
        self.seed = int(self._setting("seed", 42) if seed is None else seed)
        self.temp_root = str(temp_root or self._setting("temp_root", ".tmp"))
        self._run_id = uuid4().hex[:12]
        if self.repeats < 1:
            raise ValueError("benchmark repeats must be >= 1")

    def _need_authority(self) -> None:
        """Reject execution after the owning Context rotates its resources."""
        if self.context.modules() is not self.resolver:
            raise RuntimeError("BenchmarkRunner belongs to a retired Context authority generation")

    def _setting(self, key: str, default: Any) -> Any:
        """Return one setting from this runner's immutable config snapshot."""
        return self._settings.get(key, default)

    def _combinations(self, selected: str | Sequence[str | BackendCombination] | None) -> list[BackendCombination]:
        """Resolve caller selections through this runner's module authority."""
        if selected is None or selected == "":
            combinations = list(BackendCombination.list(resolver=self.resolver, suite="default"))
        else:
            values: Sequence[str | BackendCombination]
            if isinstance(selected, str):
                values = tuple(item.strip() for item in selected.split(",") if item.strip())
            else:
                values = selected
            combinations = [value if isinstance(value, BackendCombination) else BackendCombination.load(str(value), resolver=self.resolver) for value in values]
        if not combinations:
            raise ValueError("at least one benchmark combination is required")
        labels = [combination.label for combination in combinations]
        if len(set(labels)) != len(labels):
            raise ValueError(f"benchmark combinations repeat labels: {labels!r}")
        return combinations

    def run(self) -> BenchmarkReport:
        """Run every configured combination, scale, and repeat.

        Args:
            None.

        Returns:
            BenchmarkReport: Structured raw and aggregated benchmark results.

        Raises:
            RuntimeError: If the owning Context has closed or reset.
        """
        self._need_authority()
        wall_start = time.perf_counter()
        report = BenchmarkReport(
            meta={
                "run_id": self._run_id,
                "scenario": self.scenario.label,
                "scales": self.scales,
                "repeats": self.repeats,
                "seed": self.seed,
                "sys_info": _sys_info(),
            }
        )
        total_runs = len(self.combinations) * len(self.scales) * self.repeats
        run_idx = 0
        for combo in self.combinations:
            for scale in self.scales:
                for repeat in range(1, self.repeats + 1):
                    self._need_authority()
                    run_idx += 1
                    n = int(self.scenario.base_rows) * scale
                    print(f"[benchmark] [{run_idx}/{total_runs}] {combo.label} scale={scale}x repeat={repeat}/{self.repeats} n={n:,} ...")
                    run = self.run_one(combo, scale, repeat, n)
                    report.add(run)
                    print(f"[benchmark] [{run_idx}/{total_runs}] done in {run.total_sec:.2f}s")
        report.meta["wall_sec"] = time.perf_counter() - wall_start
        report.meta["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return report

    def run_one(self, combo: BackendCombination, scale: int, repeat: int, n: int) -> BenchRun:
        """Run one Backend combination at one scale and repeat.

        Args:
            combo (BackendCombination): Validated recipe to execute.
            scale (int): Positive workload multiplier.
            repeat (int): One-based repeat number.
            n (int): Number of scenario rows to create.

        Returns:
            BenchRun: Captured stages and resources for this run.
        """
        self._need_authority()
        ctx = self._ctx(combo, scale, repeat)
        rows = self._rows(combo, scale, repeat, n)
        ids = [row["object_id"] for row in rows]
        result = BenchRun(combo.label, combo.combination_type, {}, self.scenario.label, scale, repeat, n)
        result.resources.append(_resources("before_build", ctx, combo))
        state: dict[str, Any] = {}

        def setup() -> None:
            touch_dir(ctx.tmp, clear=True)
            backends = combo.build(ctx)
            ws = self.scenario.make_workspace(ctx, combo, backends)
            state.update({"ws": ws, "backends": ws.backends.backends})
            entity = self.scenario.make_entity(ctx, combo)
            ws.register(entity)
            state["entity"] = entity

        try:
            result.stages.append(_stage(BenchmarkOp("setup", BUCKET_BUILD, setup, notes="workspace + entity registration")))
        except BaseException:
            self._cleanup(state, ctx)
            raise
        if result.stages[-1].error_type:
            self._cleanup(state, ctx)
            return result

        result.backend_identifiers = {name: backend.type for name, backend in state["backends"].items()}
        try:
            batch_size = min(int(self._setting("max_upsert_batch", 2_000)), max(1, n // 10))
            ws = state["ws"]
            entity = state["entity"]
            result.stages.append(
                _stage(
                    BenchmarkOp(
                        "bulk_upsert",
                        BUCKET_BUILD,
                        lambda: [ws.upsert_many(entity, rows[i : i + batch_size]) for i in range(0, n, batch_size)],
                        n,
                        f"upsert_many batch={batch_size}",
                    )
                )
            )
            result.resources.append(_resources("after_build", ctx, combo))

            samples = self._samples(combo, scale, repeat, ids)
            op_ctx = ScenarioContext(ws, entity, rows, ids, samples, combo, scale, repeat, self.seed)
            for op in self.scenario.query_ops(op_ctx):
                result.stages.append(_stage(op))
            for op in self.scenario.edit_ops(op_ctx):
                result.stages.append(_stage(op))
            result.resources.append(_resources("after_edit", ctx, combo))
            for op in self.scenario.removal_ops(op_ctx):
                result.stages.append(_stage(op))
            result.resources.append(_resources("after_removal", ctx, combo))
            return result
        finally:
            self._cleanup(state, ctx)

    def _ctx(self, combo: BackendCombination, scale: int, repeat: int) -> CombinationContext:
        """Build the isolated identity for one run."""
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in combo.label).strip("_")
        ws_id = f"bench_{self._run_id}_{safe_label}_{scale}_{repeat}"
        tmp = pj(self.temp_root, f"bench-{self._run_id}-{safe_label}-{scale}-{repeat}", abs=True)
        return CombinationContext(self.context, combo.label, scale, repeat, self.seed, tmp, ws_id)

    def _rows(self, combo: BackendCombination, scale: int, repeat: int, n: int) -> list[dict[str, Any]]:
        """Build deterministic rows for one run."""
        rng = StableRNG(seed=self.seed).step(combo.label, scale, repeat, "rows")
        return self.scenario.make_rows(n, rng)

    def _samples(self, combo: BackendCombination, scale: int, repeat: int, ids: Sequence[str]) -> dict[str, list[str]]:
        """Build deterministic operation samples for one run."""
        rng = StableRNG(seed=self.seed).step(combo.label, scale, repeat, "sample")
        return {
            "get": _sample(ids, rng.step("get"), int(self._setting("batch_get_size", 1_000))),
            "exists": _sample(ids, rng.step("exists"), int(self._setting("batch_get_size", 1_000))),
            "set": _sample(ids, rng.step("set"), int(self._setting("batch_set_size", 1_000))),
            "delete": _sample(ids, rng.step("delete"), int(self._setting("batch_delete_size", 1_000))),
            "point": _sample(ids, rng.step("point"), int(self._setting("point_lookup_sample", 100))),
        }

    def _cleanup(self, state: Mapping[str, Any], ctx: CombinationContext) -> None:
        """Best-effort release of run-owned workspace and filesystem state."""
        try:
            ws = state.get("ws")
            if ws is not None:
                ws.drop()
        except Exception:
            pass
        try:
            delete_dir(ctx.tmp)
        except (FileNotFoundError, PermissionError):
            pass
