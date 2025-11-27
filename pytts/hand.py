"""Hand class for managing a player's cards."""

from typing import List, Optional
from .card import Card


class Hand:
    """
    Represents a player's hand of cards.

    A hand is a collection of cards that a player holds and can play.
    """

    def __init__(self, max_size: Optional[int] = None):
        """
        Initialize a hand.

        Args:
            max_size: Maximum number of cards allowed in hand (None for unlimited)
        """
        self.cards: List[Card] = []
        self.max_size = max_size

    def add_card(self, card: Card) -> bool:
        """
        Add a card to the hand.

        Args:
            card: The card to add

        Returns:
            True if card was added, False if hand is full
        """
        if self.max_size is not None and len(self.cards) >= self.max_size:
            return False
        self.cards.append(card)
        return True

    def remove_card(self, card: Card) -> bool:
        """
        Remove a card from the hand.

        Args:
            card: The card to remove

        Returns:
            True if card was removed, False if card not in hand
        """
        if card in self.cards:
            self.cards.remove(card)
            return True
        return False

    def get_card(self, index: int) -> Optional[Card]:
        """
        Get a card by index.

        Args:
            index: Index of the card

        Returns:
            The card at the index, or None if invalid index
        """
        if 0 <= index < len(self.cards):
            return self.cards[index]
        return None

    def find_cards(self, **properties) -> List[Card]:
        """
        Find all cards matching the given properties.

        Args:
            **properties: Properties to match

        Returns:
            List of matching cards
        """
        matching_cards = []
        for card in self.cards:
            if all(card.get_property(k) == v for k, v in properties.items()):
                matching_cards.append(card)
        return matching_cards

    def size(self) -> int:
        """Return the number of cards in the hand."""
        return len(self.cards)

    def is_empty(self) -> bool:
        """Check if the hand is empty."""
        return len(self.cards) == 0

    def is_full(self) -> bool:
        """Check if the hand is full."""
        if self.max_size is None:
            return False
        return len(self.cards) >= self.max_size

    def clear(self) -> List[Card]:
        """
        Remove all cards from the hand.

        Returns:
            List of cards that were in the hand
        """
        cards = self.cards[:]
        self.cards = []
        return cards

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Hand({len(self.cards)} cards)"

    def __str__(self) -> str:
        if not self.cards:
            return "Hand(empty)"
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"Hand([{cards_str}])"
