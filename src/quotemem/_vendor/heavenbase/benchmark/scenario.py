"""Reusable benchmark scenario definitions."""

from __future__ import annotations

__all__ = [
    "BenchmarkOp",
    "BenchmarkScenario",
    "DefaultProductScenario",
    "ScenarioContext",
]

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from heavenbase.utils.rng import StableRNG

from ..entity import Entity, field as hb_field
from ..strategies import VectorIndex
from ..types import Float, Integer, LongText, ShortText, Vector
from .combination import BackendCombination, CombinationContext
from .model import BUCKET_EDIT, BUCKET_QUERY, BUCKET_REMOVAL

if TYPE_CHECKING:
    from ..workspace import HeavenBase


@dataclass
class BenchmarkOp:
    """A runnable benchmark operation supplied by a scenario."""

    name: str
    bucket: str
    fn: Callable[[], Any]
    rows: int = 0
    notes: str = ""


@dataclass
class ScenarioContext:
    """Runtime objects available while building scenario operations."""

    ws: HeavenBase
    entity: type[Entity]
    rows: list[dict[str, Any]]
    ids: list[str]
    samples: dict[str, list[str]]
    combination: BackendCombination
    scale: int
    repeat: int
    seed: int


class BenchmarkScenario:
    """Base class for user-supplied benchmark workloads."""

    label = "scenario"
    base_rows = 1_000
    vector_dim = 1024

    def make_workspace(
        self,
        ctx: CombinationContext,
        combination: BackendCombination,
        backends: dict[str, dict[str, Any]],
    ) -> HeavenBase:
        """Build the Context-owned workspace used by one benchmark run.

        Args:
            ctx (CombinationContext): Explicit benchmark run authority and identity.
            combination (BackendCombination): Declarative Backend placement recipe.
            backends (dict[str, dict[str, Any]]): Expanded Backend construction specs.

        Returns:
            HeavenBase: Detached workspace for the run.
        """
        from ..workspace import HeavenBase

        return HeavenBase._construct(
            ctx.ws_id,
            context=ctx.context,
            backends=backends,
            detached=True,
            tracked=False,
            register_workspace=False,
            register_pending=False,
            track_live=False,
            publish=False,
        )

    def make_entity(self, ctx: CombinationContext, combination: BackendCombination) -> type[Entity]:
        """Return the entity class registered for one benchmark combination.

        Args:
            ctx (CombinationContext): Explicit benchmark run authority and identity.
            combination (BackendCombination): Declarative backend placement recipe.

        Returns:
            type[Entity]: Entity class to register for the run.
        """
        raise NotImplementedError

    def make_rows(self, n: int, rng: StableRNG) -> list[dict[str, Any]]:
        """Return deterministic input rows for the requested benchmark size.

        Args:
            n (int): Number of rows to generate.
            rng (StableRNG): Deterministic random generator for the run.

        Returns:
            list[dict[str, Any]]: Input rows in benchmark insertion order.
        """
        raise NotImplementedError

    def query_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return read/query operations to time after the build phase.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Query operations executed after build completes.
        """
        return []

    def edit_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return mutation operations to time after query operations.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Edit operations executed after query stages.
        """
        return []

    def removal_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return deletion and cleanup-validation operations to time last.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Removal operations executed after edit stages.
        """
        return []


@dataclass
class DefaultProductScenario(BenchmarkScenario):
    """Product-like scalar/vector workload used by the bundled benchmark."""

    label: str = "product"
    base_rows: int = 1_000
    vector_dim: int = 1024
    top_k: int = 10
    templates: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.templates:
            return
        self.templates = [
            {
                "name": "ANC Headphones",
                "body": "quiet audio for travel",
                "price": 129.0,
                "stock": 5,
                "status": "active",
                "category": "electronics",
                "rating": 4.5,
            },
            {"name": "Oak Desk", "body": "compact oak desk for writing", "price": 95.0, "stock": 3, "status": "active", "category": "furniture", "rating": 4.2},
            {
                "name": "Ergonomic Chair",
                "body": "office chair with lumbar support",
                "price": 249.0,
                "stock": 0,
                "status": "inactive",
                "category": "furniture",
                "rating": 4.7,
            },
            {
                "name": "USB-C Hub",
                "body": "multiport adapter for laptops",
                "price": 39.0,
                "stock": 12,
                "status": "active",
                "category": "electronics",
                "rating": 3.9,
            },
            {"name": "Monitor Stand", "body": "adjustable monitor riser", "price": 59.0, "stock": 7, "status": "active", "category": "office", "rating": 4.0},
            {
                "name": "Mechanical Keyboard",
                "body": "tactile mechanical keyboard",
                "price": 149.0,
                "stock": 2,
                "status": "active",
                "category": "electronics",
                "rating": 4.6,
            },
            {
                "name": "Webcam 4K",
                "body": "high resolution video camera",
                "price": 199.0,
                "stock": 4,
                "status": "active",
                "category": "electronics",
                "rating": 4.3,
            },
            {"name": "Desk Lamp", "body": "led desk lamp with dimmer", "price": 29.0, "stock": 20, "status": "active", "category": "office", "rating": 4.1},
        ]

    def make_entity(self, ctx: CombinationContext, combination: BackendCombination) -> type[Entity]:
        """Return the product entity routed by the active backend combination.

        Args:
            ctx (CombinationContext): Explicit benchmark run authority and identity.
            combination (BackendCombination): Declarative backend placement recipe.

        Returns:
            type[Entity]: Product-like benchmark entity class for the run.
        """
        scalar_backend = combination.scalar_backend
        vector_backend = combination.vector_backend
        dim = self.vector_dim

        class BenchItem(Entity):
            identifier = "bench-item"

            name = hb_field(ShortText).store(to=scalar_backend)
            body = hb_field(LongText).store(to=scalar_backend)
            price = hb_field(Float).store(to=scalar_backend)
            stock = hb_field(Integer).store(to=scalar_backend)
            status = hb_field(ShortText).store(to=scalar_backend)
            category = hb_field(ShortText).store(to=scalar_backend)
            rating = hb_field(Float).store(to=scalar_backend)
            emb = hb_field(Vector[dim]).store(to=vector_backend, strategy=VectorIndex)

        return BenchItem

    def make_rows(self, n: int, rng: StableRNG) -> list[dict[str, Any]]:
        """Return deterministic product rows with scalar fields and vectors.

        Args:
            n (int): Number of rows to generate.
            rng (StableRNG): Deterministic random generator for the run.

        Returns:
            list[dict[str, Any]]: Product rows including ``emb`` vectors.
        """
        rows = []
        for i in range(n):
            tmpl = self.templates[i % len(self.templates)]
            rows.append(
                {
                    "object_id": f"p{i:08d}",
                    "name": f"{tmpl['name']} {i}",
                    "body": f"{tmpl['body']} item {i}",
                    "price": tmpl["price"] + (i % 10),
                    "stock": tmpl["stock"] + (i % 5),
                    "status": tmpl["status"],
                    "category": tmpl["category"],
                    "rating": tmpl["rating"],
                }
            )
        with rng as gen:
            embs = gen.rnd_vec(dim=self.vector_dim, n=n)
        for i, row in enumerate(rows):
            row["emb"] = embs[i]
        return rows

    def query_vec(self) -> list[float]:
        """Return the deterministic vector used by bundled near queries.

        Args:
            None.

        Returns:
            list[float]: Unit vector along the first embedding dimension.
        """
        return [1.0] + [0.0] * (self.vector_dim - 1)

    def query_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return the bundled scalar, boolean, vector, and hybrid query stages.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Bundled query operations for the default scenario.
        """
        ws = ctx.ws
        entity = ctx.entity
        get_ids = ctx.samples["get"]
        exists_ids = ctx.samples["exists"]
        sample_ids = ctx.samples["point"]
        vec = self.query_vec()
        n = len(ctx.rows)
        return [
            BenchmarkOp("batch_get_many", BUCKET_QUERY, lambda: ws.get_many(get_ids, entity=entity), len(get_ids), f"get_many ({len(get_ids)})"),
            BenchmarkOp(
                "batch_exists_many", BUCKET_QUERY, lambda: ws.exists_many(exists_ids, entity=entity), len(exists_ids), f"exists_many ({len(exists_ids)})"
            ),
            BenchmarkOp("point_lookups", BUCKET_QUERY, lambda: [ws.get(row_id) for row_id in sample_ids], len(sample_ids), f"get sample={len(sample_ids)}"),
            BenchmarkOp(
                "point_get_entity",
                BUCKET_QUERY,
                lambda: [ws.get(row_id, entity=entity) for row_id in sample_ids],
                len(sample_ids),
                f"get(entity=...) sample={len(sample_ids)}",
            ),
            BenchmarkOp(
                "point_exists_entity",
                BUCKET_QUERY,
                lambda: [ws.exists(row_id, entity=entity) for row_id in sample_ids],
                len(sample_ids),
                f"exists(entity=...) sample={len(sample_ids)}",
            ),
            BenchmarkOp("operator_getitem", BUCKET_QUERY, lambda: [ws[row_id] for row_id in sample_ids], len(sample_ids), f"ws[id] sample={len(sample_ids)}"),
            BenchmarkOp(
                "operator_contains", BUCKET_QUERY, lambda: [row_id in ws for row_id in sample_ids], len(sample_ids), f"id in ws sample={len(sample_ids)}"
            ),
            BenchmarkOp("workspace_count", BUCKET_QUERY, lambda: ws.count(entity), n, "ws.count(entity)"),
            BenchmarkOp("workspace_len", BUCKET_QUERY, lambda: len(ws), n, "len(ws)"),
            BenchmarkOp("iter_workspace", BUCKET_QUERY, lambda: list(ws), n, "list(iter(ws))"),
            BenchmarkOp("ids_entity", BUCKET_QUERY, lambda: ws.ids(entity), n, "ws.ids(entity)"),
            BenchmarkOp("rows_entity", BUCKET_QUERY, lambda: ws.rows(entity), n, "ws.rows(entity)"),
            BenchmarkOp("items_entity", BUCKET_QUERY, lambda: ws.items(entity), n, "ws.items(entity)"),
            BenchmarkOp("filter_gt", BUCKET_QUERY, lambda: ws.query(entity).where(entity.price > 50.0).execute(), n, "price > 50"),
            BenchmarkOp(
                "filter_range", BUCKET_QUERY, lambda: ws.query(entity).where((entity.price >= 50.0) & (entity.price <= 150.0)).execute(), n, "price in [50,150]"
            ),
            BenchmarkOp("filter_text", BUCKET_QUERY, lambda: ws.query(entity).where(entity.body.match("audio")).execute(), n, "body.match(audio)"),
            BenchmarkOp(
                "filter_bool_tree",
                BUCKET_QUERY,
                lambda: ws.query(entity).where(((entity.price >= 50.0) | (entity.stock > 0)) & ~entity.body.match("nonexistent")).execute(),
                n,
                "bool tree",
            ),
            BenchmarkOp(
                "complex_query",
                BUCKET_QUERY,
                lambda: ws.query(entity)
                .where(
                    (entity.price >= 30.0)
                    & (entity.price <= 200.0)
                    & (entity.stock > 0)
                    & (entity.stock < 100)
                    & entity.body.match("audio")
                    & entity.name.contains("Desk")
                    & (entity.status == "active")
                    & (entity.category == "furniture")
                    & (entity.rating >= 3.0)
                    & (entity.rating <= 5.0)
                    & ~entity.body.match("nonexistent")
                )
                .execute(),
                n,
                "10+ filter complex",
            ),
            BenchmarkOp("vector_near", BUCKET_QUERY, lambda: ws.query(entity).near(entity.emb, vec, top_k=self.top_k).execute(), n, f"near top_k={self.top_k}"),
            BenchmarkOp(
                "hybrid_query",
                BUCKET_QUERY,
                lambda: ws.query(entity).where(entity.price > 50.0).near(entity.emb, vec, top_k=self.top_k).execute(),
                n,
                f"filter + near top_k={self.top_k}",
            ),
            BenchmarkOp("count_all", BUCKET_QUERY, lambda: ws.query(entity).count().execute().scalar(), n, "count"),
        ]

    def edit_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return the bundled batch patch stage.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Bundled edit operations for the default scenario.
        """
        set_ids = ctx.samples["set"]
        set_rows = [{"object_id": row_id, "price": 999.0} for row_id in set_ids]
        return [BenchmarkOp("batch_set_many", BUCKET_EDIT, lambda: ctx.ws.set_many(set_rows, entity=ctx.entity), len(set_rows), f"set_many ({len(set_rows)})")]

    def removal_ops(self, ctx: ScenarioContext) -> list[BenchmarkOp]:
        """Return the bundled batch delete and post-delete count stages.

        Args:
            ctx (ScenarioContext): Runtime workspace, entity, and sample state.

        Returns:
            list[BenchmarkOp]: Bundled removal operations for the default scenario.
        """
        del_ids = ctx.samples["delete"]
        return [
            BenchmarkOp(
                "batch_delete_many",
                BUCKET_REMOVAL,
                lambda: ctx.ws.delete_many([{"entity": ctx.entity, "object_id": row_id} for row_id in del_ids]),
                len(del_ids),
                f"delete_many ({len(del_ids)})",
            ),
            BenchmarkOp(
                "count_after_delete",
                BUCKET_REMOVAL,
                lambda: ctx.ws.query(ctx.entity).count().execute().scalar(),
                len(ctx.rows),
                "count after delete",
            ),
        ]
