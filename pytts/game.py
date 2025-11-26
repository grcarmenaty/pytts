"""Game class for orchestrating the entire game."""

from typing import List, Optional, Callable
from .player import Player
from .deck import Deck, DiscardPile
from .board import Board
from .round import Round


class Game:
    """
    The main game class that orchestrates all game elements.

    A game has players, decks, boards, and manages rounds and turns.
    """

    def __init__(self, name: str = "Game"):
        """
        Initialize a game.

        Args:
            name: The name of the game
        """
        self.name = name
        self.players: List[Player] = []
        self.decks: dict[str, Deck] = {}
        self.boards: dict[str, Board] = {}
        self.discard_piles: dict[str, DiscardPile] = {}
        self.round_number = 0
        self.turn_number = 0
        self.is_finished = False
        self.winner: Optional[Player] = None
        self._win_condition: Optional[Callable[['Game'], bool]] = None

    def add_player(self, player: Player) -> None:
        """
        Add a player to the game.

        Args:
            player: The player to add
        """
        self.players.append(player)

    def add_deck(self, name: str, deck: Deck) -> None:
        """
        Add a named deck to the game.

        Args:
            name: The name/identifier for the deck
            deck: The deck to add
        """
        self.decks[name] = deck

    def add_board(self, name: str, board: Board) -> None:
        """
        Add a named board to the game.

        Args:
            name: The name/identifier for the board
            board: The board to add
        """
        self.boards[name] = board

    def add_discard_pile(self, name: str, discard_pile: DiscardPile) -> None:
        """
        Add a named discard pile to the game.

        Args:
            name: The name/identifier for the discard pile
            discard_pile: The discard pile to add
        """
        self.discard_piles[name] = discard_pile

    def get_deck(self, name: str) -> Optional[Deck]:
        """Get a deck by name."""
        return self.decks.get(name)

    def get_board(self, name: str) -> Optional[Board]:
        """Get a board by name."""
        return self.boards.get(name)

    def get_discard_pile(self, name: str) -> Optional[DiscardPile]:
        """Get a discard pile by name."""
        return self.discard_piles.get(name)

    def set_win_condition(self, condition: Callable[['Game'], bool]) -> None:
        """
        Set the win condition for the game.

        Args:
            condition: A function that takes the game and returns True if game is won
        """
        self._win_condition = condition

    def check_win_condition(self) -> bool:
        """
        Check if the win condition has been met.

        Returns:
            True if game is won
        """
        if self._win_condition is not None:
            return self._win_condition(self)
        return False

    def get_active_players(self) -> List[Player]:
        """Get all active players."""
        return [p for p in self.players if p.is_active]

    def play_round(self) -> None:
        """Play one complete round."""
        self.round_number += 1
        round_obj = Round(self.round_number, self.players)
        round_obj.play_round(self)

    def run(
        self,
        max_rounds: Optional[int] = None,
        max_turns: Optional[int] = None
    ) -> None:
        """
        Run the game for a specified number of rounds/turns or until win condition.

        Args:
            max_rounds: Maximum number of rounds to play (None for unlimited)
            max_turns: Maximum number of total turns to play (None for unlimited)
        """
        while not self.is_finished:
            if max_rounds is not None and self.round_number >= max_rounds:
                break

            if max_turns is not None and self.turn_number >= max_turns:
                break

            active_players = self.get_active_players()
            if not active_players:
                self.is_finished = True
                break

            self.play_round()
            self.turn_number += len(active_players)

            if self.check_win_condition():
                self.is_finished = True
                break

    def end_game(self, winner: Optional[Player] = None) -> None:
        """
        End the game.

        Args:
            winner: The winning player (if any)
        """
        self.is_finished = True
        self.winner = winner

    def get_winner(self) -> Optional[Player]:
        """Get the winner of the game."""
        return self.winner

    def get_player_by_name(self, name: str) -> Optional[Player]:
        """
        Get a player by name.

        Args:
            name: The player's name

        Returns:
            The player, or None if not found
        """
        for player in self.players:
            if player.name == name:
                return player
        return None

    def reset(self) -> None:
        """Reset the game to initial state."""
        self.round_number = 0
        self.turn_number = 0
        self.is_finished = False
        self.winner = None
        for player in self.players:
            player.hand.clear()
            player.score = 0
            player.activate()
        for board in self.boards.values():
            board.clear()

    def __repr__(self) -> str:
        return (
            f"Game({self.name}, {len(self.players)} players, "
            f"round {self.round_number}, finished={self.is_finished})"
        )

    def __str__(self) -> str:
        status = "Finished" if self.is_finished else "In Progress"
        players_str = ", ".join(p.name for p in self.players)
        return (
            f"{self.name} ({status})\n"
            f"Round: {self.round_number}\n"
            f"Players: {players_str}"
        )
