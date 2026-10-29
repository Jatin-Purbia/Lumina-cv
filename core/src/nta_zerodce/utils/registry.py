"""Generic named metadata lookup; loss-specific gates live in losses/registry.

See docs/INTERFACES.md and docs/METHOD.md for the normative contract.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Typed unique-name lookup; no tensor or paper-specific computation."""

    def __init__(self) -> None:
        """Construct an empty typed registry.

        Args:
            self: Generic Registry[T].

        Returns:
            No value; owns a string-to-T mapping.

        Method:
            CONTRIBUTING registration convention.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T) -> None:
        """Register one unique name.

        Args:
            name: Nonempty unique identifier.
            value: Metadata or callable of generic type T.

        Returns:
            None; ValueError on duplicates.

        Method:
            CONTRIBUTING fail-fast duplicate policy.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        if not name or name in self._items:
            raise ValueError(f"Empty or duplicate registry key: {name}")
        self._items[name] = value

    def get(self, name: str) -> T:
        """Retrieve registered metadata.

        Args:
            name: Previously registered identifier.

        Returns:
            T; KeyError for unknown name.

        Method:
            CONTRIBUTING explicit lookup.

        Raises:
            ValueError: Invalid inputs or incompatible configuration.
        """
        return self._items[name]
