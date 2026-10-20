"""Explicit registry/factory for generic Deep Research systems."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from benchmark_harness.systems.contracts import DeepResearchSystem, SystemDescriptor


class SystemRegistryError(ValueError):
    """Raised for duplicate, unknown, or descriptor/factory mismatches."""


@dataclass(frozen=True, slots=True)
class _Registration:
    descriptor: SystemDescriptor
    factory: Callable[[], DeepResearchSystem]


class SystemRegistry:
    """The only supported construction path for system adapters."""

    def __init__(self) -> None:
        self._registrations: dict[str, _Registration] = {}

    def register(
        self,
        descriptor: SystemDescriptor,
        factory: Callable[[], DeepResearchSystem],
    ) -> None:
        if descriptor.system_id in self._registrations:
            raise SystemRegistryError(f"system {descriptor.system_id!r} is already registered.")
        if not callable(factory):
            raise SystemRegistryError("system factory must be callable.")
        self._registrations[descriptor.system_id] = _Registration(descriptor, factory)

    def descriptor(self, system_id: str) -> SystemDescriptor:
        try:
            return self._registrations[system_id].descriptor
        except KeyError as exc:
            raise SystemRegistryError(f"unknown system {system_id!r}.") from exc

    def create(self, system_id: str) -> DeepResearchSystem:
        try:
            registration = self._registrations[system_id]
        except KeyError as exc:
            raise SystemRegistryError(f"unknown system {system_id!r}.") from exc
        system = registration.factory()
        if system.descriptor != registration.descriptor:
            raise SystemRegistryError(f"factory descriptor mismatch for system {system_id!r}.")
        return system

    def descriptors(self) -> tuple[SystemDescriptor, ...]:
        return tuple(self._registrations[key].descriptor for key in sorted(self._registrations))

    def __contains__(self, system_id: str) -> bool:
        return system_id in self._registrations


def register_systems(
    registrations: Iterable[tuple[SystemDescriptor, Callable[[], DeepResearchSystem]]],
) -> SystemRegistry:
    registry = SystemRegistry()
    for descriptor, factory in registrations:
        registry.register(descriptor, factory)
    return registry
