"""Slot class for board positions."""

from typing import Optional, Callable, Any
from .card import Card


class Slot:
    """
    Represents a slot on a game board where cards can be placed.

    Slots have rules about what cards can be placed, by whom, and when.
    """

    def __init__(
        self,
        name: str,
        can_place_rule: Optional[Callable[[Card, 'Player'], bool]] = None,
        can_take_rule: Optional[Callable[[Card, 'Player'], bool]] = None,
        max_cards: int = 1
    ):
        """
        Initialize a slot.

        Args:
            name: The name/identifier of the slot
            can_place_rule: Function(card, player) -> bool to determine if placement is allowed
            can_take_rule: Function(card, player) -> bool to determine if taking is allowed
            max_cards: Maximum number of cards that can be in this slot
        """
        self.name = name
        self.cards: list[Card] = []
        self.max_cards = max_cards
        self._can_place_rule = can_place_rule
        self._can_take_rule = can_take_rule
        self.properties: dict[str, Any] = {}

    def set_property(self, key: str, value: Any) -> None:
        """Set a property of the slot."""
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property of the slot."""
        return self.properties.get(key, default)

    def can_place_card(self, card: Card, player: 'Player') -> bool:
        """
        Check if a card can be placed in this slot by a player.

        Args:
            card: The card to place
            player: The player attempting to place the card

        Returns:
            True if placement is allowed
        """
        if len(self.cards) >= self.max_cards:
            return False

        if self._can_place_rule is not None:
            return self._can_place_rule(card, player)

        return True

    def can_take_card(self, card: Card, player: 'Player') -> bool:
        """
        Check if a card can be taken from this slot by a player.

        Args:
            card: The card to take
            player: The player attempting to take the card

        Returns:
            True if taking is allowed
        """
        if card not in self.cards:
            return False

        if self._can_take_rule is not None:
            return self._can_take_rule(card, player)

        return True

    def place_card(self, card: Card, player: 'Player') -> bool:
        """
        Place a card in this slot.

        Args:
            card: The card to place
            player: The player placing the card

        Returns:
            True if successful
        """
        if not self.can_place_card(card, player):
            return False

        self.cards.append(card)
        card.current_slot = self
        return True

    def remove_card(self, card: Card) -> bool:
        """
        Remove a card from this slot.

        Args:
            card: The card to remove

        Returns:
            True if successful
        """
        if card in self.cards:
            self.cards.remove(card)
            if card.current_slot == self:
                card.current_slot = None
            return True
        return False

    def get_cards(self) -> list[Card]:
        """Get all cards in this slot."""
        return self.cards[:]

    def is_empty(self) -> bool:
        """Check if the slot is empty."""
        return len(self.cards) == 0

    def is_full(self) -> bool:
        """Check if the slot is full."""
        return len(self.cards) >= self.max_cards

    def clear(self) -> list[Card]:
        """
        Remove all cards from the slot.

        Returns:
            List of cards that were in the slot
        """
        cards = self.cards[:]
        for card in cards:
            if card.current_slot == self:
                card.current_slot = None
        self.cards = []
        return cards

    def __repr__(self) -> str:
        return f"Slot({self.name}, {len(self.cards)}/{self.max_cards} cards)"

    def __str__(self) -> str:
        if not self.cards:
            return f"{self.name} (empty)"
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"{self.name} [{cards_str}]"
