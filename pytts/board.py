"""Board class for managing game board."""

from typing import List, Optional, Dict
from .slot import Slot


class Board:
    """
    Represents a game board containing slots where cards can be placed.

    The board is a collection of slots, each with their own rules.
    """

    def __init__(self, name: str = "Board"):
        """
        Initialize a board.

        Args:
            name: The name of the board
        """
        self.name = name
        self.slots: List[Slot] = []
        self._slot_map: Dict[str, Slot] = {}

    def add_slot(self, slot: Slot) -> None:
        """
        Add a slot to the board.

        Args:
            slot: The slot to add
        """
        self.slots.append(slot)
        self._slot_map[slot.name] = slot

    def get_slot(self, name: str) -> Optional[Slot]:
        """
        Get a slot by name.

        Args:
            name: The name of the slot

        Returns:
            The slot, or None if not found
        """
        return self._slot_map.get(name)

    def get_slot_by_index(self, index: int) -> Optional[Slot]:
        """
        Get a slot by index.

        Args:
            index: The index of the slot

        Returns:
            The slot, or None if invalid index
        """
        if 0 <= index < len(self.slots):
            return self.slots[index]
        return None

    def remove_slot(self, slot: Slot) -> bool:
        """
        Remove a slot from the board.

        Args:
            slot: The slot to remove

        Returns:
            True if successful
        """
        if slot in self.slots:
            self.slots.remove(slot)
            if slot.name in self._slot_map:
                del self._slot_map[slot.name]
            return True
        return False

    def get_all_slots(self) -> List[Slot]:
        """Get all slots on the board."""
        return self.slots[:]

    def get_empty_slots(self) -> List[Slot]:
        """Get all empty slots on the board."""
        return [slot for slot in self.slots if slot.is_empty()]

    def get_occupied_slots(self) -> List[Slot]:
        """Get all occupied slots on the board."""
        return [slot for slot in self.slots if not slot.is_empty()]

    def clear(self) -> None:
        """Clear all cards from all slots on the board."""
        for slot in self.slots:
            slot.clear()

    def __repr__(self) -> str:
        return f"Board({self.name}, {len(self.slots)} slots)"

    def __str__(self) -> str:
        if not self.slots:
            return f"{self.name} (no slots)"
        slots_str = "\n  ".join(str(slot) for slot in self.slots)
        return f"{self.name}:\n  {slots_str}"
