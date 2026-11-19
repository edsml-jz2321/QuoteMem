"""Compose benchmark HTTP operations with initialized application owners.

This adapter supplies no lifecycle, Search implementation, or recovery owner.
The composition root must provide those dependencies before serving requests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from aml.api.bench import AddDispatch, SearchDelivery, SearchRequest
from aml.core.add_outcome import AddOutcomeV1
from aml.core.errors import AMLError
from aml.ingestion.boundary import AddRequestV1
from aml.ingestion.service import AddService
from aml.service.lifecycle import ServiceLifetime, ServiceUnavailable

type SearchOperation = Callable[[SearchRequest, SearchDelivery], None]


class BenchService:
    """Adapt the real reception owner and required guarded Search operation.

    The required lifecycle protects verification and whole application calls
    against resource shutdown. Its composition owner supplies readiness checks
    and drains listeners/workers before closing resources. Search retains the
    guarded delivery obligations of BenchmarkApplication.
    """

    def __init__(
        self,
        reception: AddService,
        *,
        search: SearchOperation,
        lifetime: ServiceLifetime,
    ) -> None:
        """Bind existing application dependencies without starting them.

        Args:
            reception: Real Add service configured for the benchmark profile.
            search: Required operation that delivers under the final read guard.
            lifetime: Required admission owner verifying every serving dependency.

        Returns:
            None.

        Raises:
            AMLError: If reception is configured for a different profile.
        """
        if reception.profile != "bench":
            raise AMLError("service_unhealthy", "Benchmark reception requires the bench profile")
        self._reception = reception
        self._search = search
        self._lifetime = lifetime

    @contextmanager
    def add_reception(self, request: AddRequestV1) -> Iterator[AddDispatch]:
        """Reserve parsed Add scheduling before its joined application worker starts.

        Args:
            request: The validated input with its original logical identity.

        Yields:
            The reception's cooperative control while its whole lifetime is admitted.
        """
        try:
            with self._lifetime.admission():
                self._require_bench()
                with self._reception.add_reception(request) as dispatch:
                    yield dispatch
        except ServiceUnavailable as error:
            raise AMLError(
                "service_unhealthy", "Benchmark service is not accepting work"
            ) from error

    def add(self, request: AddRequestV1) -> AddOutcomeV1:
        """Acknowledge only after the existing reception owner returns.

        Args:
            request: The admitted HTTP request DTO, passed directly to AddService.

        Returns:
            The immutable result of this completed request.

        Raises:
            AMLError: If readiness is withdrawn or the reception owner rejects work.
        """
        try:
            with self._lifetime.admission():
                self._require_bench()
                receipt = self._reception.add(request)
                if receipt.outcome is None:
                    raise AMLError("incomplete_coverage", "Add has no completed request result")
                return receipt.outcome
        except ServiceUnavailable as error:
            raise AMLError(
                "service_unhealthy", "Benchmark service is not accepting work"
            ) from error

    def search(self, request: SearchRequest, delivery: SearchDelivery) -> None:
        """Delegate the exact request and transport control to the required use case.

        Args:
            request: The admitted exact-user Search request.
            delivery: Transport deadline/cancellation and guarded response enqueue owner.

        Returns:
            None after the Search operation settles its complete delivery lifetime.

        Raises:
            AMLError: If readiness is withdrawn or the Search operation rejects work.
        """
        try:
            with self._lifetime.admission():
                self._require_bench()
                self._search(request, delivery)
        except ServiceUnavailable as error:
            raise AMLError(
                "service_unhealthy", "Benchmark service is not accepting work"
            ) from error

    def health(self) -> bool:
        """Report the required lifecycle owner's current readiness.

        Args:
            None.

        Returns:
            True only for a benchmark reception owner and an explicitly ready lifecycle.
        """
        return self._reception.profile == "bench" and self._lifetime.healthy()

    def _require_bench(self) -> None:
        if self._reception.profile != "bench":
            raise AMLError("service_unhealthy", "Benchmark reception requires the bench profile")
