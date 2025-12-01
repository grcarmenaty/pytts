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

    def _score_room_bonus(self, work: 'Card', rooms: dict) -> float:
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

        return best_room_score

    def _score_fashion_match(self, work: 'Card', player: 'ModernismePlayer', game: 'ModernismeGame') -> float:
        """Score how well a work matches fashion objectives (moda tema and conjunto)."""
        score = 0.0
        board = game.get_board(f"{player.name}_board")
        if not board:
            return 0.0

        # Score moda tema match
        if game.moda_tema:
            required_theme = game.moda_tema.get_property("required_theme")
            if work.get_property("theme") == required_theme:
                # Count current matching works
                theme_count = sum(1 for slot in board.slots
                                if not slot.is_empty() and
                                slot.get_cards()[0].get_property("theme") == required_theme)
                # More valuable if we're close to completing (need 3)
                if theme_count >= 2:
                    score += 3.0  # Very close to completion
                elif theme_count >= 1:
                    score += 2.0  # Making progress
                else:
                    score += 1.0  # Starting the objective

        # Score moda conjunto match
        if game.moda_conjunto:
            required_themes = game.moda_conjunto.get_property("required_themes")
            work_theme = work.get_property("theme")
            if work_theme in required_themes:
                # Count works of each required theme
                theme_counts = {theme: 0 for theme in required_themes}
                for slot in board.slots:
                    if not slot.is_empty():
                        slot_theme = slot.get_cards()[0].get_property("theme")
                        if slot_theme in required_themes:
                            theme_counts[slot_theme] += 1

                # Prefer themes we have less of
                current_count = theme_counts.get(work_theme, 0)
                min_count = min(theme_counts.values())
                if current_count == min_count:
                    score += 2.0  # Helps balance the set
                else:
                    score += 1.0  # Still useful

        return score

    def _score_commission_match(self, work: 'Card', player: 'ModernismePlayer', game: 'ModernismeGame') -> float:
        """Score how well a work matches the player's commission objective."""
        if not player.commission_card:
            return 0.0

        score = 0.0
        encargo = player.commission_card
        obj_type = encargo.get_property("objective_type")
        board = game.get_board(f"{player.name}_board")

        if not board:
            return 0.0

        work_type = work.get_property("art_type")

        # Count current works by type
        type_counts = {}
        for slot in board.slots:
            if not slot.is_empty():
                slot_work = slot.get_cards()[0]
                slot_type = slot_work.get_property("art_type")
                type_counts[slot_type] = type_counts.get(slot_type, 0) + 1

        if obj_type == "type_count":
            required_type = encargo.get_property("required_type")
            required_count = encargo.get_property("required_count", 3)
            if work_type == required_type:
                current = type_counts.get(required_type, 0)
                if current >= required_count - 1:
                    score += 3.0  # Very close to completion
                elif current >= required_count - 2:
                    score += 2.0  # Making progress
                else:
                    score += 1.0  # Starting the objective

        elif obj_type == "mixed":
            required_types = encargo.get_property("required_types")
            required_counts = encargo.get_property("required_counts")

            for req_type, req_count in zip(required_types, required_counts):
                if work_type == req_type:
                    current = type_counts.get(req_type, 0)
                    progress = current / req_count
                    # More valuable if this type needs more progress
                    if progress < 0.5:
                        score += 2.0
                    elif progress < 1.0:
                        score += 1.5
                    else:
                        score += 0.5  # Already complete

        elif obj_type == "theme_count":
            required_theme = encargo.get_property("required_theme")
            if work.get_property("theme") == required_theme:
                theme_count = sum(1 for slot in board.slots
                                if not slot.is_empty() and
                                slot.get_cards()[0].get_property("theme") == required_theme)
                if theme_count >= 2:
                    score += 3.0
                elif theme_count >= 1:
                    score += 2.0
                else:
                    score += 1.0

        return score

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

        board = game.get_board(f"{player.name}_board")
        rooms = self._analyze_rooms(board) if board else {}

        # Score each work: primary = low VP, secondary = room/fashion/commission bonuses
        best_work = None
        best_score = float('inf')

        for work in commissionable:
            # Primary: VP cost (lower is better, so we use regular value)
            vp_cost = work.get_property("vp", 0)

            # Secondary factors (these become tiebreakers for similar VP)
            room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
            fashion_score = self._score_fashion_match(work, player, game)
            commission_score = self._score_commission_match(work, player, game)

            # Combined score: prioritize low VP, use secondary factors as tiebreakers
            # Multiply VP by 100 to make it dominant, subtract bonuses to prefer higher bonuses
            total_score = (vp_cost * 100.0) - (room_score + fashion_score + commission_score)

            if total_score < best_score:
                best_score = total_score
                best_work = work

        return best_work


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

        board = game.get_board(f"{player.name}_board")
        rooms = self._analyze_rooms(board) if board else {}

        # Score each work: primary = high VP, secondary = room/fashion/commission bonuses
        best_work = None
        best_score = -float('inf')

        for work in commissionable:
            # Primary: VP value (higher is better)
            vp_value = work.get_property("vp", 0)

            # Secondary factors (these become tiebreakers for similar VP)
            room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
            fashion_score = self._score_fashion_match(work, player, game)
            commission_score = self._score_commission_match(work, player, game)

            # Combined score: prioritize high VP, use secondary factors as tiebreakers
            # Multiply VP by 100 to make it dominant, add bonuses
            total_score = (vp_value * 100.0) + room_score + fashion_score + commission_score

            if total_score > best_score:
                best_score = total_score
                best_work = work

        return best_work


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

        board = game.get_board(f"{player.name}_board")
        rooms = self._analyze_rooms(board) if board else {}

        # Get the required theme from moda_tema
        if not game.moda_tema:
            return random.choice(commissionable)

        required_theme = game.moda_tema.get_property("required_theme")

        # Prioritize works with the required theme
        matching_works = [w for w in commissionable if w.get_property("theme") == required_theme]

        if matching_works:
            # Among matching works, consider VP, room bonuses, and commission objectives
            best_work = None
            best_score = -float('inf')

            for work in matching_works:
                vp_value = work.get_property("vp", 0)
                room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                commission_score = self._score_commission_match(work, player, game)

                # Primary: matching theme (already filtered), secondary: VP + room + commission
                total_score = (vp_value * 10.0) + room_score + commission_score

                if total_score > best_score:
                    best_score = total_score
                    best_work = work

            return best_work
        else:
            # If no matching works, be competent: consider room bonuses while saving resources
            best_work = None
            best_score = float('inf')

            for work in commissionable:
                vp_cost = work.get_property("vp", 0)
                room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                commission_score = self._score_commission_match(work, player, game)

                # Prefer low VP, but consider room and commission bonuses
                total_score = (vp_cost * 10.0) - (room_score + commission_score)

                if total_score < best_score:
                    best_score = total_score
                    best_work = work

            return best_work


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

        board = game.get_board(f"{player.name}_board")
        rooms = self._analyze_rooms(board) if board else {}

        # Get the required themes from moda_conjunto
        if not game.moda_conjunto:
            return random.choice(commissionable)

        required_themes = game.moda_conjunto.get_property("required_themes")

        # Count current works of each required theme
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
            # Among matching works, consider VP, room bonuses, and commission
            best_work = None
            best_score = -float('inf')

            for work in matching_works:
                vp_value = work.get_property("vp", 0)
                room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                commission_score = self._score_commission_match(work, player, game)

                # Primary: needed theme (already filtered), secondary: VP + room + commission
                total_score = (vp_value * 10.0) + room_score + commission_score

                if total_score > best_score:
                    best_score = total_score
                    best_work = work

            return best_work
        else:
            # Try other required themes with competent scoring
            for theme in required_themes:
                theme_works = [w for w in commissionable if w.get_property("theme") == theme]
                if theme_works:
                    best_work = None
                    best_score = -float('inf')

                    for work in theme_works:
                        vp_value = work.get_property("vp", 0)
                        room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                        commission_score = self._score_commission_match(work, player, game)

                        total_score = (vp_value * 10.0) + room_score + commission_score

                        if total_score > best_score:
                            best_score = total_score
                            best_work = work

                    return best_work

            # If no matching works, be competent: consider room bonuses while saving resources
            best_work = None
            best_score = float('inf')

            for work in commissionable:
                vp_cost = work.get_property("vp", 0)
                room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                commission_score = self._score_commission_match(work, player, game)

                # Prefer low VP, but consider room and commission bonuses
                total_score = (vp_cost * 10.0) - (room_score + commission_score)

                if total_score < best_score:
                    best_score = total_score
                    best_work = work

            return best_work


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

        board = game.get_board(f"{player.name}_board")
        rooms = self._analyze_rooms(board) if board else {}

        # Get the commission card
        if not player.commission_card:
            return random.choice(commissionable)

        encargo = player.commission_card
        obj_type = encargo.get_property("objective_type")

        # Count current works by type
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
                # Among matching works, consider VP, room bonuses, and fashion objectives
                best_work = None
                best_score = -float('inf')

                for work in matching_works:
                    vp_value = work.get_property("vp", 0)
                    room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                    fashion_score = self._score_fashion_match(work, player, game)

                    # Primary: commission type (already filtered), secondary: VP + room + fashion
                    total_score = (vp_value * 10.0) + room_score + fashion_score

                    if total_score > best_score:
                        best_score = total_score
                        best_work = work

                return best_work

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
                # Among matching works, consider VP, room bonuses, and fashion objectives
                best_work = None
                best_score = -float('inf')

                for work in matching_works:
                    vp_value = work.get_property("vp", 0)
                    room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                    fashion_score = self._score_fashion_match(work, player, game)

                    total_score = (vp_value * 10.0) + room_score + fashion_score

                    if total_score > best_score:
                        best_score = total_score
                        best_work = work

                return best_work

        elif obj_type == "theme_count":
            required_theme = encargo.get_property("required_theme")
            matching_works = [w for w in commissionable if w.get_property("theme") == required_theme]
            if matching_works:
                best_work = None
                best_score = -float('inf')

                for work in matching_works:
                    vp_value = work.get_property("vp", 0)
                    room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
                    fashion_score = self._score_fashion_match(work, player, game)

                    total_score = (vp_value * 10.0) + room_score + fashion_score

                    if total_score > best_score:
                        best_score = total_score
                        best_work = work

                return best_work

        # Fallback: choose work with best overall score
        best_work = None
        best_score = -float('inf')

        for work in commissionable:
            vp_value = work.get_property("vp", 0)
            room_score = self._score_room_bonus(work, rooms) if rooms else 0.0
            fashion_score = self._score_fashion_match(work, player, game)

            total_score = (vp_value * 10.0) + room_score + fashion_score

            if total_score > best_score:
                best_score = total_score
                best_work = work

        return best_work


class RandomStrategy(Strategy):
    """Random strategy - selects works randomly as a baseline for comparison."""

    def __init__(self):
        super().__init__("Random")

    def select_work(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame'
    ) -> Optional['Card']:
        """Select a random commissionable work."""
        commissionable = self.get_commissionable_works(player)
        if not commissionable:
            return None

        return random.choice(commissionable)


# List of all available strategies
ALL_STRATEGIES = [
    RoomThemeTypeStrategy,
    MaxWorksStrategy,
    HighValueWorksStrategy,
    ModaTemaStrategy,
    ModaConjuntoStrategy,
    EncargoStrategy,
    RandomStrategy
]


def get_random_strategy() -> Strategy:
    """Get a random strategy instance."""
    strategy_class = random.choice(ALL_STRATEGIES)
    return strategy_class()
