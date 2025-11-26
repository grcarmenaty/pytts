"""Deck classes for managing card collections."""

import random
from typing import List, Optional
from .card import Card
from .hand import Hand


class Deck:
    """
    Represents a deck of cards.

    A deck manages a collection of cards and supports drawing and shuffling.
    """

    def __init__(self, cards: Optional[List[Card]] = None):
        """
        Initialize a deck.

        Args:
            cards: Initial list of cards (default: empty deck)
        """
        self.cards: List[Card] = cards[:] if cards else []
        self._visible_to_all = False

    def add_card(self, card: Card) -> None:
        """Add a card to the top of the deck."""
        self.cards.append(card)

    def draw(self, count: int = 1) -> List[Card]:
        """
        Draw cards from the top of the deck.

        Args:
            count: Number of cards to draw

        Returns:
            List of drawn cards (may be fewer than requested if deck is small)
        """
        drawn = []
        for _ in range(min(count, len(self.cards))):
            drawn.append(self.cards.pop())
        return drawn

    def draw_to_hand(self, hand: Hand, count: int = 1) -> int:
        """
        Draw cards from the deck to a player's hand.

        Args:
            hand: The hand to draw cards to
            count: Number of cards to draw

        Returns:
            Number of cards actually drawn
        """
        drawn_count = 0
        for _ in range(count):
            if not self.cards:
                break
            if hand.is_full():
                break
            card = self.cards.pop()
            if hand.add_card(card):
                drawn_count += 1
            else:
                self.cards.append(card)
                break
        return drawn_count

    def shuffle(self) -> None:
        """Shuffle the deck randomly."""
        random.shuffle(self.cards)

    def peek(self, count: int = 1) -> List[Card]:
        """
        Look at the top cards without drawing them.

        Args:
            count: Number of cards to peek at

        Returns:
            List of cards from the top (may be fewer than requested)
        """
        return self.cards[-count:] if self.cards else []

    def size(self) -> int:
        """Return the number of cards in the deck."""
        return len(self.cards)

    def is_empty(self) -> bool:
        """Check if the deck is empty."""
        return len(self.cards) == 0

    def is_visible_to_all(self) -> bool:
        """Check if all cards in this deck are visible to all players."""
        return self._visible_to_all

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f"Deck({len(self.cards)} cards)"


class DiscardPile(Deck):
    """
    A special type of deck where all cards are visible to all players.

    Typically used for discarded cards that all players can see.
    """

    def __init__(self, cards: Optional[List[Card]] = None):
        """
        Initialize a discard pile.

        Args:
            cards: Initial list of cards (default: empty)
        """
        super().__init__(cards)
        self._visible_to_all = True

    def get_all_cards(self) -> List[Card]:
        """
        Get all cards in the discard pile (visible to all players).

        Returns:
            List of all cards in the pile
        """
        return self.cards[:]

    def peek_all(self) -> List[Card]:
        """
        View all cards in the discard pile.

        Returns:
            List of all cards (same as get_all_cards)
        """
        return self.get_all_cards()

    def __repr__(self) -> str:
        return f"DiscardPile({len(self.cards)} cards)"
