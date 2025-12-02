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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """
        Select which room tile to take from the available tiles.

        Args:
            player: The player making the decision
            game: The game instance
            available_tiles: List of available room tiles

        Returns:
            The room tile to select, or None if none should be selected
        """
        # Default: random selection
        if not available_tiles:
            return None
        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """
        Select which advantage card to take from the available cards.

        Args:
            player: The player making the decision
            game: The game instance
            available_cards: List of available advantage cards

        Returns:
            The advantage card to select, or None if none should be selected
        """
        # Default: random selection
        if not available_cards:
            return None
        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """
        Decide whether to acquire a room tile for 2 VP.

        Args:
            player: The player making the decision
            game: The game instance
            room_name: The room that needs a tile

        Returns:
            True if should acquire the tile, False otherwise
        """
        # Default: acquire if we have enough cards and need it
        return len(player.hand.cards) >= 3

    def select_room_for_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        tile: 'Card',
        available_rooms: List[str]
    ) -> Optional[str]:
        """
        Select which room to assign a newly acquired tile to.

        Args:
            player: The player making the decision
            game: The game instance
            tile: The room tile being placed
            available_rooms: List of room names that don't have tiles yet

        Returns:
            The room name to assign the tile to, or None for random
        """
        # Default: prefer smaller rooms (easier to complete)
        if not available_rooms:
            return None

        # Parse room sizes from names like "Room 1 (3 slots)"
        room_sizes = []
        for room in available_rooms:
            try:
                size = int(room.split('(')[1].split()[0])
                room_sizes.append((room, size))
            except:
                room_sizes.append((room, 999))  # Unknown size, deprioritize

        # Sort by size (smallest first)
        room_sizes.sort(key=lambda x: x[1])
        return room_sizes[0][0]

    def should_use_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        phase: str
    ) -> Optional['Card']:
        """
        Decide whether to use an advantage card at this moment.

        Args:
            player: The player making the decision
            game: The game instance
            phase: Current phase ('before_turn', 'during_turn', 'after_commission')

        Returns:
            The advantage card to use, or None if no card should be used
        """
        # Default: don't use advantage cards automatically (too complex for default)
        return None

    def select_artist_to_hire(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_artists: List['Card']
    ) -> 'Card':
        """
        Select which artist to hire from the available options.

        Args:
            player: The player making the decision
            game: The game instance
            available_artists: List of artists drawn from the deck

        Returns:
            The artist to hire
        """
        # Default: select artist that matches most works in hand
        if not available_artists:
            return available_artists[0]

        best_artist = available_artists[0]
        best_match_count = 0

        for artist in available_artists:
            artist_type = artist.get_property("art_type")
            match_count = sum(1 for work in player.hand.cards
                            if work.get_property("art_type") == artist_type)
            if match_count > best_match_count:
                best_match_count = match_count
                best_artist = artist

        return best_artist

    def select_artist_to_dismiss(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        active_artists: List['Card'],
        hired_artist: 'Card'
    ) -> 'Card':
        """
        Select which active artist to dismiss when hiring a new one.

        Args:
            player: The player making the decision
            game: The game instance
            active_artists: List of currently active artists
            hired_artist: The artist being hired

        Returns:
            The artist to dismiss
        """
        # Default: dismiss the artist with fewest matching works
        if len(active_artists) < 2:
            return active_artists[0]

        artist_match_counts = []
        for artist in active_artists:
            artist_type = artist.get_property("art_type")
            match_count = sum(1 for work in player.hand.cards
                            if work.get_property("art_type") == artist_type)
            artist_match_counts.append((artist, match_count))

        # Sort by match count (ascending) and dismiss the one with least matches
        artist_match_counts.sort(key=lambda x: x[1])
        return artist_match_counts[0][0]

    def steal_artist_from_neighbor(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_artists: List['Card']
    ) -> Optional['Card']:
        """
        Decide whether to steal an artist from a neighbor.

        Args:
            player: The player making the decision
            game: The game instance
            available_artists: List of artists available to steal

        Returns:
            The artist to steal, or None if shouldn't steal
        """
        # Default: steal if there's an artist that matches many works in hand
        if not available_artists:
            return None

        best_artist = None
        best_match_count = 2  # Only steal if at least 2 matches

        for artist in available_artists:
            artist_type = artist.get_property("art_type")
            match_count = sum(1 for work in player.hand.cards
                            if work.get_property("art_type") == artist_type)
            if match_count > best_match_count:
                best_match_count = match_count
                best_artist = artist

        return best_artist

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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Select room tile that best supports room completion strategy."""
        if not available_tiles:
            return None

        # Prefer tiles that match our current work distribution
        board = game.get_board(f"{player.name}_board")
        if not board:
            return random.choice(available_tiles)

        # Count works by type and theme
        type_counts = {}
        theme_counts = {}
        for slot in board.slots:
            if not slot.is_empty():
                work = slot.get_cards()[0]
                art_type = work.get_property("art_type")
                theme = work.get_property("theme")
                type_counts[art_type] = type_counts.get(art_type, 0) + 1
                if theme and hasattr(theme, 'value') and theme.value != "No theme":
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Score each tile
        best_tile = None
        best_score = -1

        for tile in available_tiles:
            tile_type = tile.get_property("tile_type")
            is_theme = tile.get_property("is_theme_tile", False)
            score = 0

            if is_theme:
                # Theme tile - check if we have works with this theme
                for theme, count in theme_counts.items():
                    if hasattr(theme, 'value') and tile_type.value == theme.value:
                        score = count * 2  # Prefer themes we already have
            else:
                # Type tile - check if we have works with this type
                for art_type, count in type_counts.items():
                    if hasattr(art_type, 'value') and tile_type.value == art_type.value:
                        score = count * 2

            if score > best_score:
                best_score = score
                best_tile = tile

        return best_tile if best_tile else random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select advantage card that helps with room completion."""
        if not available_cards:
            return None

        # Prefer cards that help rearrange or refresh rooms
        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.REMODELING,  # Move works to optimize rooms
            AdvantageCardType.REFORM,      # Refresh room tiles
            AdvantageCardType.PLAN_CHANGE, # Return work to hand if misplaced
            AdvantageCardType.PATRONAGE,   # Change artists for flexibility
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Acquire room tiles strategically based on game state."""
        # Don't acquire if we're low on cards
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        # Count current usable slots and commissionable works
        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        commissionable_works = self.get_commissionable_works(player)

        # If we have plenty of usable slots relative to our works, be conservative
        if usable_slots >= len(commissionable_works) + 2:
            return False

        # If we're running low on slots, acquire
        if usable_slots <= 1:
            return True

        # Otherwise, moderate acquisition for room completion strategy
        return len(player.hand.cards) >= 4

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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Select type tiles for common art types to maximize placement options."""
        if not available_tiles:
            return None

        # Prefer type tiles over theme tiles (more flexible)
        type_tiles = [t for t in available_tiles if not t.get_property("is_theme_tile", False)]
        if type_tiles:
            # Prefer common types (Crafts, Painting, Sculpture over Relic)
            from examples.modernisme_advanced import RoomTileType
            preferred_types = [RoomTileType.CRAFTS, RoomTileType.PAINTING, RoomTileType.SCULPTURE]

            for pref_type in preferred_types:
                for tile in type_tiles:
                    if tile.get_property("tile_type") == pref_type:
                        return tile

            return type_tiles[0]

        # If only theme tiles available, pick randomly
        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select cards that help place more works."""
        if not available_cards:
            return None

        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.UNIVERSAL_EXHIBITION,  # Redraw hand for better options
            AdvantageCardType.PATRONAGE,            # Change artists to commission more
            AdvantageCardType.REFORM,               # More room tiles = more placement options
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Acquire room tiles to unlock more placement spaces."""
        # Need at least moderate hand size
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        commissionable_works = self.get_commissionable_works(player)

        # MaxWorks is more aggressive - wants space for all works
        # If we have plenty of slots relative to our works, be conservative
        if usable_slots >= len(commissionable_works) + 3:
            return False

        # If we're running low on slots, acquire
        if usable_slots <= 2:
            return True

        # Acquire if we have many works and moderate slots
        return len(commissionable_works) >= 3 and len(player.hand.cards) >= 4


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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Prefer type tiles for high-VP art types."""
        if not available_tiles:
            return None

        # High VP works are often Relics (5 VP) or high-value regular works (3-4 VP)
        # Prefer type tiles, especially for common high-value types
        type_tiles = [t for t in available_tiles if not t.get_property("is_theme_tile", False)]
        if type_tiles:
            from examples.modernisme_advanced import RoomTileType
            # Relic tiles are good (all relics are 5 VP)
            for tile in type_tiles:
                if tile.get_property("tile_type") == RoomTileType.RELIC:
                    return tile
            # Otherwise any type tile
            return type_tiles[0]

        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select cards that help place high-value works."""
        if not available_cards:
            return None

        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.PATRONAGE,     # Change artists to get high-VP matches
            AdvantageCardType.REMODELING,    # Optimize placement for room bonuses
            AdvantageCardType.UNIVERSAL_EXHIBITION,  # Redraw for better cards
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Strategic room tile acquisition for high-VP strategy."""
        # Don't acquire if hand is too small
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        # Count high-VP commissionable works
        commissionable_works = self.get_commissionable_works(player)
        high_vp_works = [w for w in commissionable_works if w.get_property("vp", 0) >= 3]

        # If we have plenty of slots relative to works, be conservative
        if usable_slots >= len(commissionable_works) + 2:
            return False

        # Prioritize acquiring if we have high-VP works to place
        if len(high_vp_works) >= 2 and usable_slots <= 2:
            return True

        # If running low on slots, acquire
        if usable_slots <= 1:
            return True

        # Otherwise be conservative (high VP works are scarce)
        return len(player.hand.cards) >= 5


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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Prefer theme tiles matching moda tema."""
        if not available_tiles:
            return None

        # Strongly prefer theme tile matching the moda tema
        if hasattr(game, 'moda_tema') and game.moda_tema:
            required_theme = game.moda_tema.get_property("required_theme")
            if required_theme and hasattr(required_theme, 'value'):
                for tile in available_tiles:
                    if tile.get_property("is_theme_tile", False):
                        tile_type = tile.get_property("tile_type")
                        if hasattr(tile_type, 'value') and tile_type.value == required_theme.value:
                            return tile

        # Otherwise prefer any theme tile
        theme_tiles = [t for t in available_tiles if t.get_property("is_theme_tile", False)]
        if theme_tiles:
            return theme_tiles[0]

        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select cards that help collect theme works."""
        if not available_cards:
            return None

        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.UNIVERSAL_EXHIBITION,  # Redraw to find more theme matches
            AdvantageCardType.PATRONAGE,            # Change artists for theme matches
            AdvantageCardType.ESPIONAGE,            # Steal theme cards from opponents
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Moderate acquisition - focus on getting theme-matching tiles."""
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        commissionable_works = self.get_commissionable_works(player)

        # If we have plenty of slots relative to works, be conservative
        if usable_slots >= len(commissionable_works) + 2:
            return False

        # If running low, acquire
        if usable_slots <= 1:
            return True

        # Moderate acquisition
        return len(player.hand.cards) >= 4


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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Prefer theme tiles for required themes in moda conjunto."""
        if not available_tiles:
            return None

        # Prefer theme tiles for the required themes
        if hasattr(game, 'moda_conjunto') and game.moda_conjunto:
            required_themes = game.moda_conjunto.get_property("required_themes")
            if required_themes:
                for tile in available_tiles:
                    if tile.get_property("is_theme_tile", False):
                        tile_type = tile.get_property("tile_type")
                        for req_theme in required_themes:
                            if hasattr(tile_type, 'value') and hasattr(req_theme, 'value'):
                                if tile_type.value == req_theme.value:
                                    return tile

        # Otherwise prefer any theme tile
        theme_tiles = [t for t in available_tiles if t.get_property("is_theme_tile", False)]
        if theme_tiles:
            return theme_tiles[0]

        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select cards that help create adjacent sets."""
        if not available_cards:
            return None

        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.REMODELING,            # Move works to create adjacencies
            AdvantageCardType.UNIVERSAL_EXHIBITION,  # Redraw for required themes
            AdvantageCardType.PATRONAGE,            # Change artists for theme matches
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Moderate acquisition for set fashion strategy."""
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        commissionable_works = self.get_commissionable_works(player)

        # If we have plenty of slots relative to works, be conservative
        if usable_slots >= len(commissionable_works) + 2:
            return False

        # If running low, acquire
        if usable_slots <= 1:
            return True

        # Moderate acquisition
        return len(player.hand.cards) >= 4


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

    def select_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_tiles: List['Card']
    ) -> Optional['Card']:
        """Select type tiles matching commission requirements."""
        if not available_tiles:
            return None

        # Prefer type tiles for required types in commission
        if player.commission_card:
            encargo = player.commission_card
            obj_type = encargo.get_property("objective_type")

            # Get required types
            required_types = []
            if obj_type == "type_count":
                required_types = [encargo.get_property("required_type")]
            elif obj_type == "mixed":
                required_types = encargo.get_property("required_types", [])

            # Find tiles matching required types
            for req_type in required_types:
                for tile in available_tiles:
                    if not tile.get_property("is_theme_tile", False):
                        tile_type = tile.get_property("tile_type")
                        if hasattr(tile_type, 'value') and hasattr(req_type, 'value'):
                            if tile_type.value == req_type.value:
                                return tile

        # Otherwise prefer any type tile
        type_tiles = [t for t in available_tiles if not t.get_property("is_theme_tile", False)]
        if type_tiles:
            return type_tiles[0]

        return random.choice(available_tiles)

    def select_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        available_cards: List['Card']
    ) -> Optional['Card']:
        """Select cards that help complete commission."""
        if not available_cards:
            return None

        from examples.modernisme_advanced import AdvantageCardType

        priority_order = [
            AdvantageCardType.PATRONAGE,            # Change artists to match commission needs
            AdvantageCardType.UNIVERSAL_EXHIBITION,  # Redraw for required types
            AdvantageCardType.ESPIONAGE,            # Steal needed cards
        ]

        for adv_type in priority_order:
            for card in available_cards:
                if card.get_property("advantage_type") == adv_type:
                    return card

        return random.choice(available_cards)

    def should_acquire_room_tile(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        room_name: str
    ) -> bool:
        """Moderate acquisition for commission strategy."""
        if len(player.hand.cards) < 3:
            return False

        board = game.get_board(f"{player.name}_board")
        if not board:
            return False

        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in getattr(player, 'room_tiles', {}))

        commissionable_works = self.get_commissionable_works(player)

        # If we have plenty of slots relative to works, be conservative
        if usable_slots >= len(commissionable_works) + 2:
            return False

        # If running low, acquire
        if usable_slots <= 1:
            return True

        # Moderate acquisition
        return len(player.hand.cards) >= 4


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

    def should_use_advantage_card(
        self,
        player: 'ModernismePlayer',
        game: 'ModernismeGame',
        phase: str
    ) -> Optional['Card']:
        """Use advantage cards randomly."""
        if not player.advantage_cards:
            return None

        # 20% chance to use a card at any given opportunity
        if random.random() < 0.2:
            return random.choice(player.advantage_cards)

        return None


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
