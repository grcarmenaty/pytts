"""Strategy classes for AI players in Modernisme game."""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING, Tuple
import random

if TYPE_CHECKING:
    from .card import Card
    from examples.modernisme_game import ModernismePlayer, ModernismeGame, ArtType, Theme


class Strategy(ABC):
    """Abstract base class for player strategies."""

    def __init__(self, name: str):
        """
        Initialize a strategy.

        Args:
            name: The name of the strategy
        """
        self.name = name

    @abstractmethod
    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """
        Select which work card to commission from the player's hand.

        Args:
            player: The player making the decision
            game: The game instance

        Returns:
            The work card to commission, or None if no work should be commissioned
        """
        pass

    def get_commissionable_works(
        self,
        player: 'ModernismePlayer'
    ) -> List['Card']:
        """
        Get all works that can be commissioned (player has matching artist).

        Args:
            player: The player

        Returns:
            List of commissionable work cards
        """
        commissionable = []
        for work in player.hand.cards:
            work_type = work.get_property("art_type")
            can_commission = any(
                artist.get_property("art_type") == work_type
                for artist in player.active_artists
            )
            if can_commission:
                commissionable.append(work)
        return commissionable

    def __str__(self) -> str:
        return self.name


class RoomThemeTypeStrategy(Strategy):
    """
    Strategy focusing on completing rooms with same theme/different types
    or same type/different themes for bonus VP.
    """

    def __init__(self):
        super().__init__("Room Theme/Type Optimizer")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select work that best completes room bonuses."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        board = game.get_board(f"{player.name}_board")
        if not board:
            return random.choice(commissionable)

        # Analyze current room state
        rooms = self._analyze_rooms(board)

        # Score each work based on how well it contributes to room bonuses
        best_work = None
        best_score = -float('inf')

        for work in commissionable:
            score = self._score_work_for_rooms(work, rooms)
            if score > best_score:
                best_score = score
                best_work = work

        return best_work if best_work else random.choice(commissionable)

    def _analyze_rooms(self, board) -> dict:
        """Analyze the current state of all rooms."""
        from examples.modernisme_game import Theme

        rooms = {}
        for slot in board.slots:
            room_name = slot.get_property("room")
            if room_name not in rooms:
                rooms[room_name] = {
                    "works": [],
                    "types": set(),
                    "themes": set(),
                    "spaces_filled": 0,
                    "total_spaces": 0
                }
            rooms[room_name]["total_spaces"] += 1
            if not slot.is_empty():
                work = slot.get_cards()[0]
                rooms[room_name]["works"].append(work)
                rooms[room_name]["types"].add(work.get_property("art_type"))
                theme = work.get_property("theme")
                if theme != Theme.NONE:
                    rooms[room_name]["themes"].add(theme)
                rooms[room_name]["spaces_filled"] += 1

        return rooms

    def _score_work_for_rooms(self, work: 'Card', rooms: dict) -> float:
        """Score a work based on its contribution to room bonuses."""
        from examples.modernisme_game import Theme

        work_type = work.get_property("art_type")
        work_theme = work.get_property("theme")

        best_room_score = 0.0

        for room_name, room_info in rooms.items():
            if room_info["spaces_filled"] >= room_info["total_spaces"]:
                continue  # Room is full

            score = 0.0

            # Check if this work maintains "same type" path
            if len(room_info["types"]) == 0:
                score += 2.0  # First work in room
            elif len(room_info["types"]) == 1 and work_type in room_info["types"]:
                score += 5.0  # Continues same type bonus
            elif len(room_info["types"]) == 1 and work_type not in room_info["types"]:
                score += 0.5  # Breaks same type, but may enable same theme

            # Check if this work maintains "same theme" path
            if work_theme != Theme.NONE:
                if len(room_info["themes"]) == 0:
                    score += 2.0
                elif len(room_info["themes"]) == 1 and work_theme in room_info["themes"]:
                    score += 5.0  # Continues same theme bonus
                elif len(room_info["themes"]) == 1 and work_theme not in room_info["themes"]:
                    score += 0.5  # Breaks same theme

            # Bonus for nearly complete rooms
            spaces_remaining = room_info["total_spaces"] - room_info["spaces_filled"]
            if spaces_remaining == 1:
                score += 10.0  # Completing a room is very valuable

            best_room_score = max(best_room_score, score)

        # Add base VP value as tiebreaker
        best_room_score += work.get_property("vp", 0) * 0.1

        return best_room_score


class MaxWorksStrategy(Strategy):
    """Strategy focusing on playing as many works as possible."""

    def __init__(self):
        super().__init__("Maximum Works")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select the lowest VP work to maximize number of works placed."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        # Choose the work with lowest VP cost (easiest to commission)
        return min(commissionable, key=lambda w: w.get_property("vp", 0))


class HighValueWorksStrategy(Strategy):
    """Strategy focusing on playing high value works."""

    def __init__(self):
        super().__init__("High Value Works")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select the highest VP work available."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        # Choose the work with highest VP
        return max(commissionable, key=lambda w: w.get_property("vp", 0))


class ModaTemaStrategy(Strategy):
    """Strategy focusing on completing moda tema (theme fashion) objectives."""

    def __init__(self):
        super().__init__("Theme Fashion Focus")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select work that best advances theme fashion objective."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        # Get the required theme from moda_tema
        if not game.moda_tema:
            return random.choice(commissionable)

        required_theme = game.moda_tema.get_property("required_theme")

        # Count current works of the required theme
        board = game.get_board(f"{player.name}_board")
        theme_count = 0
        if board:
            for slot in board.slots:
                if not slot.is_empty():
                    work = slot.get_cards()[0]
                    if work.get_property("theme") == required_theme:
                        theme_count += 1

        # Prioritize works with the required theme
        matching_works = [w for w in commissionable if w.get_property("theme") == required_theme]

        if matching_works:
            # Among matching works, prefer higher VP
            return max(matching_works, key=lambda w: w.get_property("vp", 0))
        else:
            # If no matching works, choose lowest VP to save resources
            return min(commissionable, key=lambda w: w.get_property("vp", 0))


class ModaConjuntoStrategy(Strategy):
    """Strategy focusing on completing moda conjunto (adjacent set fashion) objectives."""

    def __init__(self):
        super().__init__("Set Fashion Focus")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select work that best advances adjacent set fashion objective."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        # Get the required themes from moda_conjunto
        if not game.moda_conjunto:
            return random.choice(commissionable)

        required_themes = game.moda_conjunto.get_property("required_themes")

        # Count current works of each required theme
        board = game.get_board(f"{player.name}_board")
        theme_counts = {theme: 0 for theme in required_themes}

        if board:
            for slot in board.slots:
                if not slot.is_empty():
                    work = slot.get_cards()[0]
                    work_theme = work.get_property("theme")
                    if work_theme in required_themes:
                        theme_counts[work_theme] += 1

        # Find the theme we need most (have least of)
        needed_theme = min(theme_counts, key=theme_counts.get)

        # Prioritize works with the needed theme
        matching_works = [w for w in commissionable if w.get_property("theme") == needed_theme]

        if matching_works:
            # Among matching works, prefer higher VP
            return max(matching_works, key=lambda w: w.get_property("vp", 0))
        else:
            # Try other required themes
            for theme in required_themes:
                matching_works = [w for w in commissionable if w.get_property("theme") == theme]
                if matching_works:
                    return max(matching_works, key=lambda w: w.get_property("vp", 0))

            # If no matching works, choose lowest VP
            return min(commissionable, key=lambda w: w.get_property("vp", 0))


class EncargoStrategy(Strategy):
    """Strategy focusing on completing encargo (commission) objectives."""

    def __init__(self):
        super().__init__("Commission Focus")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select work that best advances commission objective."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        # Get the commission card
        if not player.commission_card:
            return random.choice(commissionable)

        encargo = player.commission_card
        obj_type = encargo.get_property("objective_type")

        # Count current works by type
        board = game.get_board(f"{player.name}_board")
        type_counts = {}

        if board:
            for slot in board.slots:
                if not slot.is_empty():
                    work = slot.get_cards()[0]
                    work_type = work.get_property("art_type")
                    type_counts[work_type] = type_counts.get(work_type, 0) + 1

        # Determine which type we need most based on objective
        if obj_type == "type_count":
            required_type = encargo.get_property("required_type")
            matching_works = [w for w in commissionable if w.get_property("art_type") == required_type]
            if matching_works:
                return max(matching_works, key=lambda w: w.get_property("vp", 0))

        elif obj_type == "mixed":
            required_types = encargo.get_property("required_types")
            required_counts = encargo.get_property("required_counts")

            # Find which type we need to work on
            type_progress = []
            for req_type, req_count in zip(required_types, required_counts):
                current = type_counts.get(req_type, 0)
                progress = current / req_count
                type_progress.append((req_type, progress))

            # Focus on type with least progress
            needed_type = min(type_progress, key=lambda x: x[1])[0]

            matching_works = [w for w in commissionable if w.get_property("art_type") == needed_type]
            if matching_works:
                return max(matching_works, key=lambda w: w.get_property("vp", 0))

        # Fallback: choose highest VP work
        return max(commissionable, key=lambda w: w.get_property("vp", 0))


# List of all available strategies
ALL_STRATEGIES = [
    RoomThemeTypeStrategy,
    MaxWorksStrategy,
    HighValueWorksStrategy,
    ModaTemaStrategy,
    ModaConjuntoStrategy,
    EncargoStrategy
]


def get_random_strategy() -> Strategy:
    """Get a random strategy instance."""
    strategy_class = random.choice(ALL_STRATEGIES)
    return strategy_class()
