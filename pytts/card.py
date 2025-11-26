"""Card class for tabletop simulator."""

from typing import Any, Dict, Optional


class Card:
    """
    Represents a card in the game.

    Cards have properties (attributes) and can be placed in slots on the board.
    """

    def __init__(self, name: str, **properties):
        """
        Initialize a card with a name and optional properties.

        Args:
            name: The name/identifier of the card
            **properties: Additional properties (e.g., value, suit, color, etc.)
        """
        self.name = name
        self.properties = properties
        self.current_slot: Optional['Slot'] = None
        self.owner: Optional['Player'] = None

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value by key."""
        return self.properties.get(key, default)

    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        self.properties[key] = value

    def place_in_slot(self, slot: 'Slot', player: 'Player') -> bool:
        """
        Attempt to place this card in a slot.

        Args:
            slot: The slot to place the card in
            player: The player attempting to place the card

        Returns:
            True if successful, False otherwise
        """
        if slot.can_place_card(self, player):
            if self.current_slot:
                self.current_slot.remove_card(self)
            slot.place_card(self, player)
            self.current_slot = slot
            return True
        return False

    def remove_from_slot(self, player: 'Player') -> bool:
        """
        Attempt to remove this card from its current slot.

        Args:
            player: The player attempting to remove the card

        Returns:
            True if successful, False otherwise
        """
        if self.current_slot and self.current_slot.can_take_card(self, player):
            self.current_slot.remove_card(self)
            self.current_slot = None
            return True
        return False

    def __repr__(self) -> str:
        props = ", ".join(f"{k}={v}" for k, v in self.properties.items())
        if props:
            return f"Card({self.name}, {props})"
        return f"Card({self.name})"

    def __str__(self) -> str:
        return self.name
