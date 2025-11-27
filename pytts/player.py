"""Player class with strategy pattern."""

from typing import Optional, TYPE_CHECKING
from .hand import Hand

if TYPE_CHECKING:
    from .game import Game


class Player:
    """
    Represents a player in the game.

    Players have a hand of cards and implement a strategy for playing.
    The strategy method is called during each turn to determine the player's actions.
    """

    def __init__(self, name: str, hand_max_size: Optional[int] = None):
        """
        Initialize a player.

        Args:
            name: The player's name/identifier
            hand_max_size: Maximum hand size (None for unlimited)
        """
        self.name = name
        self.hand = Hand(max_size=hand_max_size)
        self.score = 0
        self.is_active = True
        self.pool = 0  # VP pool for current turn

    def strategy(self, game: 'Game') -> None:
        """
        Execute the player's strategy for this turn.

        This method should be overridden in subclasses to implement
        specific game logic and decision-making.

        Args:
            game: The game instance providing context for decision-making
        """
        pass

    def draw_cards(self, deck: 'Deck', count: int = 1) -> int:
        """
        Draw cards from a deck to this player's hand.

        Args:
            deck: The deck to draw from
            count: Number of cards to draw

        Returns:
            Number of cards actually drawn
        """
        from .deck import Deck
        return deck.draw_to_hand(self.hand, count)

    def play_card(self, card: 'Card', slot: 'Slot') -> bool:
        """
        Play a card from hand to a slot on the board.

        Args:
            card: The card to play
            slot: The slot to place the card in

        Returns:
            True if successful
        """
        if card not in self.hand.cards:
            return False

        if card.place_in_slot(slot, self):
            self.hand.remove_card(card)
            return True
        return False

    def add_score(self, points: int) -> None:
        """
        Add points to the player's score.

        Args:
            points: Points to add (can be negative)
        """
        self.score += points

    def deactivate(self) -> None:
        """Mark the player as inactive (e.g., eliminated from game)."""
        self.is_active = False

    def activate(self) -> None:
        """Mark the player as active."""
        self.is_active = True

    def __repr__(self) -> str:
        return f"Player({self.name}, score={self.score}, cards={len(self.hand)})"

    def __str__(self) -> str:
        return f"{self.name} (score: {self.score}, hand: {len(self.hand)} cards)"
