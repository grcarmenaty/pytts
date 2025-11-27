"""Round class for managing game rounds."""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .player import Player
    from .game import Game


class Round:
    """
    Represents a round in the game.

    A round executes turns for all active players in order.
    """

    def __init__(self, round_number: int, players: List['Player']):
        """
        Initialize a round.

        Args:
            round_number: The round number
            players: List of players in turn order
        """
        self.round_number = round_number
        self.players = players
        self.current_turn = 0

    def execute_turn(self, player: 'Player', game: 'Game') -> None:
        """
        Execute a single turn for a player.

        Args:
            player: The player whose turn it is
            game: The game instance
        """
        if player.is_active:
            player.strategy(game)

    def play_round(self, game: 'Game') -> None:
        """
        Play a complete round (one turn for each active player).

        Args:
            game: The game instance
        """
        for i, player in enumerate(self.players):
            self.current_turn = i
            if player.is_active:
                self.execute_turn(player, game)

    def get_active_players(self) -> List['Player']:
        """Get all active players in this round."""
        return [p for p in self.players if p.is_active]

    def __repr__(self) -> str:
        active_count = len(self.get_active_players())
        return f"Round({self.round_number}, {active_count}/{len(self.players)} active players)"
