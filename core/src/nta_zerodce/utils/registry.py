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
            None.
        """
        self._items: dict[str, T] = {}

    def register(self, name: str, value: T) -> None:
        """Register one unique name.

        Args:
            name: Nonempty unique identifier.
            value: Metadata or callable of generic type T.

        Returns:
            None; ValueError on duplicates or empty names.

        Method:
            CONTRIBUTING fail-fast duplicate policy.

        Raises:
            ValueError: The name is not a nonempty string or is already registered.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"registry name must be a nonempty string; got {name!r}")
        if name in self._items:
            raise ValueError(f"{name!r} is already registered")
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
            KeyError: The name is not registered (the message lists the known names).
        """
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(self.names()) or "<none>"
            raise KeyError(f"unknown name {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        """List registered names.

        Args:
            self: Registry.

        Returns:
            Names in sorted order (a new list).

        Method:
            Sorted keys for deterministic messages.

        Raises:
            None.
        """
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        """Report whether a name is registered."""
        return name in self._items

    def __len__(self) -> int:
        """Return the number of registered names."""
        return len(self._items)
