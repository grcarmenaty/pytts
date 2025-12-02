"""
Modernisme Advanced - Advanced game mode with room tiles and advantage cards.

This implements the advanced rules from v0.4.1, including:
- Room tiles that must be placed before works can be placed in rooms
- Advantage cards that provide special abilities
- Variable hand sizes by season
- Enhanced room bonuses with doubling conditions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Game, Player, Card, Deck, Board, Slot, Hand
from pytts.strategy import Strategy, get_random_strategy
import random
from typing import List, Optional, Tuple, TextIO
from enum import Enum
from io import StringIO


class ArtType(Enum):
    """Types of art/artists in the game."""
    CRAFTS = "Crafts"
    PAINTING = "Painting"
    SCULPTURE = "Sculpture"
    RELIC = "Relic"


class Theme(Enum):
    """Themes for works and artists."""
    NATURE = "Nature"
    MYTHOLOGY = "Mythology"
    SOCIETY = "Society"
    ORIENTALISM = "Orientalism"
    NONE = "No theme"


class RoomTileType(Enum):
    """Types for room tiles - can be either an art type or a theme."""
    # Art types
    CRAFTS = "Crafts"
    PAINTING = "Painting"
    SCULPTURE = "Sculpture"
    RELIC = "Relic"
    # Themes
    NATURE = "Nature"
    MYTHOLOGY = "Mythology"
    SOCIETY = "Society"
    ORIENTALISM = "Orientalism"


class AdvantageCardType(Enum):
    """Types of advantage cards."""
    UNIVERSAL_EXHIBITION = "Exposició Universal"  # Redraw hand
    PATRONAGE = "Mecenatge"  # Change both artists
    CHURCH_VISIT = "Visita a la Lletja"  # Rearrange artist discard pile
    IDEA_EXCHANGE = "Intercanvi d'Idees"  # Trade cards with opponent
    ESPIONAGE = "Espionatge"  # Take from opponent's discard
    PLAN_CHANGE = "Canvi de plans"  # Return work to hand
    REMODELING = "Remodelació"  # Move two placed works
    REFORM = "Reforma"  # Refresh room tiles


class ModernismeAdvancedPlayer(Player):
    """Player in advanced Modernisme game."""

    def __init__(self, name: str, strategy: Optional[Strategy] = None):
        super().__init__(name, hand_max_size=None)
        self.active_artists: List[Card] = []
        self.discard_pile: List[Card] = []
        self.commission_card: Optional[Card] = None
        self.completed_rooms: List[int] = []
        self.ai_strategy: Optional[Strategy] = strategy
        self.reliquia_themes: dict = {}

        # Advanced mode specific
        self.room_tiles: dict = {}  # Maps room name to room tile card
        self.advantage_cards: List[Card] = []  # Advantage cards held
        self.vp_milestones: List[int] = [8, 18, 28, 40]  # VP levels that trigger advantage card selection
        self.last_milestone: int = 0  # Track last milestone passed

        # Statistics tracking
        self.turn_stats: List[dict] = []
        self.current_turn_stats: dict = {}

    def start_turn(self, season: int, turn_number: int):
        """Initialize statistics tracking for a new turn."""
        from collections import defaultdict
        self.current_turn_stats = {
            'season': season,
            'turn': turn_number,
            'vp_start': self.score,
            'vp_gained': 0,
            'cards_played': 0,
            'cards_discarded': 0,
            'cards_in_hand_start': len(self.hand.cards),
            'cards_in_hand_end': 0,
            'works_by_type': defaultdict(int),
            'works_by_theme': defaultdict(int),
            'total_vp_spent': 0,
            'total_vp_earned_from_plays': 0,
            'room_tiles_acquired': 0,
        }

    def end_turn(self):
        """Finalize statistics for the current turn."""
        self.current_turn_stats['vp_end'] = self.score
        self.current_turn_stats['vp_gained'] = self.score - self.current_turn_stats['vp_start']
        self.current_turn_stats['cards_in_hand_end'] = len(self.hand.cards)
        self.current_turn_stats['works_by_type'] = dict(self.current_turn_stats['works_by_type'])
        self.current_turn_stats['works_by_theme'] = dict(self.current_turn_stats['works_by_theme'])
        self.turn_stats.append(self.current_turn_stats.copy())
        self.current_turn_stats = {}

    def check_milestone(self, game: 'ModernismeAdvancedGame') -> bool:
        """Check if player has passed a VP milestone and should pick an advantage card."""
        for milestone in self.vp_milestones:
            if self.score >= milestone > self.last_milestone:
                self.last_milestone = milestone
                return True
        return False

    def can_place_in_room(self, room_name: str) -> bool:
        """Check if player has a room tile for the given room."""
        return room_name in self.room_tiles

    def strategy(self, game: 'ModernismeAdvancedGame') -> None:
        """AI strategy for advanced Modernisme."""
        board = game.get_board(f"{self.name}_board")

        # First, identify ALL works in hand that can be commissioned
        commissionable_works = []
        for work in self.hand.cards:
            work_type = work.get_property("art_type")
            can_commission = any(
                artist.get_property("art_type") == work_type
                for artist in self.active_artists
            )
            if can_commission:
                commissionable_works.append(work)

        # Count usable slots (empty slots in rooms we have tiles for)
        usable_slots = sum(1 for slot in board.slots
                         if slot.is_empty() and
                         slot.get_property("room") in self.room_tiles)

        # ONLY acquire room tile if we're BLOCKED (have commissionable works but NO usable slots)
        if commissionable_works and usable_slots == 0:
            # We're completely blocked! Find a room without a tile
            rooms_without_tiles = set()
            for slot in board.slots:
                if slot.is_empty():
                    room_name = slot.get_property("room")
                    if room_name not in self.room_tiles:
                        rooms_without_tiles.add(room_name)

            if rooms_without_tiles:
                room_to_acquire = list(rooms_without_tiles)[0]
                if self.ai_strategy:
                    should_acquire = self.ai_strategy.should_acquire_room_tile(self, game, room_to_acquire)
                else:
                    should_acquire = len(self.hand.cards) >= 2

                if should_acquire and self._try_acquire_room_tile(game, room_to_acquire):
                    game.log(f"  Acquired room tile (was blocked with {len(commissionable_works)} commissionable works)")

        # Now try to commission up to 2 works
        works_to_commission = min(2, len(self.hand.cards))
        for _ in range(works_to_commission):
            if not self.hand.cards:
                break

            if self.ai_strategy:
                work = self.ai_strategy.select_work(self, game)
            else:
                work = None
                for w in self.hand.cards:
                    work_type = w.get_property("art_type")
                    can_commission = any(
                        artist.get_property("art_type") == work_type
                        for artist in self.active_artists
                    )
                    if can_commission:
                        work = w
                        break

            if work and self._try_commission_work(work, game):
                continue
            else:
                break

        if self.pool > 0:
            game.log(f"  {self.pool} VP remaining in pool discarded at end of turn")
        self.pool = 0

    def _try_acquire_room_tile(self, game: 'ModernismeAdvancedGame', room_name: str) -> bool:
        """Try to acquire a room tile for 2 VP."""
        # Find cards totaling at least 2 VP
        available_cards = self.hand.cards[:]
        card_vps = [(card, card.get_property("vp", 0)) for card in available_cards]
        card_vps.sort(key=lambda x: x[1])

        # Find minimal combination that gives 2 VP
        for i in range(len(card_vps)):
            if card_vps[i][1] >= 2:
                # Single card is enough
                card = card_vps[i][0]
                self.hand.remove_card(card)
                self.discard_pile.append(card)

                # Pick a room tile
                tile = game.pick_room_tile_for_player(self, room_name)
                if tile:
                    self.room_tiles[room_name] = tile
                    if self.current_turn_stats:
                        self.current_turn_stats['room_tiles_acquired'] += 1
                    return True
                return False

            # Try combination
            total_vp = 0
            discard_cards = []
            for j in range(i, len(card_vps)):
                card, vp = card_vps[j]
                discard_cards.append(card)
                total_vp += vp
                if total_vp >= 2:
                    # Discard these cards
                    for c in discard_cards:
                        self.hand.remove_card(c)
                        self.discard_pile.append(c)

                    # Pick a room tile
                    tile = game.pick_room_tile_for_player(self, room_name)
                    if tile:
                        self.room_tiles[room_name] = tile
                        if self.current_turn_stats:
                            self.current_turn_stats['room_tiles_acquired'] += 1
                        return True
                    return False

        return False

    def _try_commission_work(self, work: Card, game: 'ModernismeAdvancedGame') -> bool:
        """Try to commission a specific work."""
        work_vp = work.get_property("vp", 0)
        vp_needed = work_vp - self.pool

        if vp_needed > 0:
            available_cards = [c for c in self.hand.cards if c != work]
            if not available_cards:
                return False

            card_vps = [(card, card.get_property("vp", 0)) for card in available_cards]
            card_vps.sort(key=lambda x: x[1])

            best_discard = None
            best_overspend = float('inf')

            for i in range(len(card_vps)):
                discard_cards = []
                total_vp = 0

                for j in range(i, len(card_vps)):
                    card, vp = card_vps[j]
                    discard_cards.append(card)
                    total_vp += vp

                    if total_vp >= vp_needed:
                        overspend = total_vp - vp_needed
                        if overspend < best_overspend:
                            best_overspend = overspend
                            best_discard = discard_cards[:]
                        break

            if best_discard is None:
                return False

            discard_vp = sum(card.get_property("vp", 0) for card in best_discard)
            discard_names = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in best_discard]
            game.log(f"    Discarded: {', '.join(discard_names)} (added {discard_vp} VP to pool)")

            for card in best_discard:
                self.hand.remove_card(card)
                self.discard_pile.append(card)
                self.pool += card.get_property("vp", 0)

                if self.current_turn_stats:
                    self.current_turn_stats['cards_discarded'] += 1

        if self.pool < work_vp:
            return False

        game.log(f"    Pool before commission: {self.pool} VP")
        self.pool -= work_vp
        game.log(f"    Spent {work_vp} VP from pool, remaining: {self.pool} VP")
        self.hand.remove_card(work)

        # Find first available room slot WITH a room tile
        board = game.get_board(f"{self.name}_board")
        if board:
            for slot in board.get_empty_slots():
                room_name = slot.get_property("room")
                # In advanced mode, can only place if room tile exists
                if room_name in self.room_tiles and slot.can_place_card(work, self):
                    slot.place_card(work, self)

                    base_vp = work_vp
                    bonus_vp = 0

                    work_theme = work.get_property("theme")
                    matching_artist = None
                    for artist in self.active_artists:
                        if artist.get_property("art_type") == work.get_property("art_type"):
                            if artist.get_property("theme") == work_theme and work_theme != Theme.NONE:
                                bonus_vp = 1
                                matching_artist = artist
                                break

                    vp_gain = base_vp + bonus_vp
                    self.add_score(vp_gain)

                    if self.current_turn_stats:
                        self.current_turn_stats['cards_played'] += 1
                        self.current_turn_stats['total_vp_spent'] += work_vp
                        self.current_turn_stats['total_vp_earned_from_plays'] += vp_gain

                        art_type = work.get_property("art_type")
                        type_key = art_type.value if hasattr(art_type, 'value') else str(art_type)
                        self.current_turn_stats['works_by_type'][type_key] += 1

                        theme = work.get_property("theme")
                        theme_key = theme.value if hasattr(theme, 'value') else str(theme)
                        self.current_turn_stats['works_by_theme'][theme_key] += 1

                    space_id = slot.get_property("space_id")
                    game.log(f"    → Placed '{work.name}' in {room_name} (space {space_id})")
                    bonus_msg = f" + {bonus_vp} (theme bonus from {matching_artist.name})" if bonus_vp > 0 else ""
                    game.log(f"      Base VP: {base_vp}{bonus_msg} = {vp_gain} VP total")
                    game.log(f"      {self.name}'s total score: {self.score} VP")

                    # Check for milestone
                    if self.check_milestone(game):
                        game.log(f"      {self.name} passed a VP milestone!")
                        game.handle_milestone_reward(self)

                    return True

        return False

    def assign_reliquia_themes(self, game: 'ModernismeAdvancedGame') -> None:
        """Assign optimal themes to all placed Reliquias."""
        board = game.get_board(f"{self.name}_board")
        if not board:
            return

        reliquias = []
        for slot in board.slots:
            if not slot.is_empty():
                work = slot.get_cards()[0]
                if work.get_property("art_type") == ArtType.RELIC:
                    reliquias.append((work, slot))

        if not reliquias:
            return

        game.log(f"\n{self.name} contextualizing Relics:")

        for work, slot in reliquias:
            best_theme = self._find_best_theme_for_reliquia(work, slot, game)
            self.reliquia_themes[work] = best_theme
            game.log(f"  {work.name} → {best_theme.value} theme")

    def _find_best_theme_for_reliquia(self, reliquia: Card, slot: 'Slot', game: 'ModernismeAdvancedGame') -> Theme:
        """Find the best theme to assign to a Reliquia."""
        board = game.get_board(f"{self.name}_board")
        room_name = slot.get_property("room")

        theme_scores = {theme: 0 for theme in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}

        # Check which theme would help with room bonus
        for theme in theme_scores.keys():
            temp_theme_count = {t: 0 for t in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}
            temp_type_count = {t: 0 for t in [ArtType.CRAFTS, ArtType.PAINTING, ArtType.SCULPTURE, ArtType.RELIC]}
            room_works = []

            for s in board.slots:
                if s.get_property("room") == room_name and not s.is_empty():
                    w = s.get_cards()[0]
                    room_works.append(w)

                    # Count type
                    temp_type_count[w.get_property("art_type")] += 1

                    # Count theme
                    if w == reliquia:
                        temp_theme_count[theme] += 1
                    elif w.get_property("art_type") == ArtType.RELIC and w in self.reliquia_themes:
                        temp_theme_count[self.reliquia_themes[w]] += 1
                    elif w.get_property("theme") != Theme.NONE:
                        temp_theme_count[w.get_property("theme")] += 1

            room_size = sum(1 for s in board.slots if s.get_property("room") == room_name)
            if len(room_works) == room_size and room_name in self.room_tiles:
                room_tile = self.room_tiles[room_name]
                tile_type = room_tile.get_property("tile_type")

                # Check for same-theme bonus
                for t, count in temp_theme_count.items():
                    if count == room_size and t == theme:
                        # Check if tile type matches
                        if tile_type in [RoomTileType.NATURE, RoomTileType.MYTHOLOGY,
                                       RoomTileType.SOCIETY, RoomTileType.ORIENTALISM]:
                            if tile_type.value == theme.value:
                                # Check if all types different (doubles bonus)
                                types_present = sum(1 for c in temp_type_count.values() if c > 0)
                                if types_present == room_size:
                                    theme_scores[theme] += room_size * 4  # Doubled
                                else:
                                    theme_scores[theme] += room_size * 2

        # Check which theme helps with moda_tema
        if game.moda_tema:
            required_theme = game.moda_tema.get_property("required_theme")
            if required_theme in theme_scores:
                theme_scores[required_theme] += 3

        # Balance themes for moda conjunto
        theme_counts = {t: 0 for t in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}
        for s in board.slots:
            if not s.is_empty():
                w = s.get_cards()[0]
                if w != reliquia:
                    if w.get_property("art_type") == ArtType.RELIC and w in self.reliquia_themes:
                        theme_counts[self.reliquia_themes[w]] += 1
                    elif w.get_property("theme") != Theme.NONE:
                        theme_counts[w.get_property("theme")] += 1

        min_count = min(theme_counts.values())
        for theme in theme_scores.keys():
            if theme_counts[theme] == min_count:
                theme_scores[theme] += 1

        best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        return best_theme


class ModernismeAdvancedGame(Game):
    """Advanced Modernisme game implementation."""

    def __init__(self, log_file: Optional[TextIO] = None):
        super().__init__("Modernisme Advanced")
        self.artist_discard: List[Card] = []
        self.season = 1
        self.first_player_idx = 0
        self.moda_tema: Optional[Card] = None
        self.moda_conjunto: Optional[Card] = None
        self.log_file = log_file

        # Advanced mode specific
        self.available_room_tiles: List[Card] = []  # Tiles visible on board
        self.room_tile_bag: List[Card] = []  # Tiles in bag
        self.advantage_deck: List[Card] = []  # Advantage card deck
        self.available_advantages: List[Card] = []  # Visible advantage cards

    def log(self, message: str = ""):
        """Log a message to file or console."""
        if self.log_file:
            self.log_file.write(message + "\n")
        else:
            print(message)

    def get_game_data(self) -> dict:
        """Get structured game data for CSV export."""
        data = {
            'game_id': id(self),
            'mode': 'advanced',
            'moda_tema': self.moda_tema.name if self.moda_tema else '',
            'moda_conjunto': self.moda_conjunto.name if self.moda_conjunto else '',
        }

        for i, player in enumerate(self.players):
            position = i + 1
            board = self.get_board(f"{player.name}_board")
            total_works = sum(1 for slot in board.slots if not slot.is_empty()) if board else 0

            data[f'p{position}_name'] = player.name
            data[f'p{position}_strategy'] = player.ai_strategy.name if player.ai_strategy else 'None'
            data[f'p{position}_final_score'] = player.score
            data[f'p{position}_total_works'] = total_works
            data[f'p{position}_commission'] = player.commission_card.name if player.commission_card else ''
            data[f'p{position}_room_tiles'] = len(player.room_tiles)

            total_cards_played = sum(turn.get('cards_played', 0) for turn in player.turn_stats)
            total_cards_discarded = sum(turn.get('cards_discarded', 0) for turn in player.turn_stats)
            total_vp_earned = sum(turn.get('total_vp_earned_from_plays', 0) for turn in player.turn_stats)
            total_vp_spent = sum(turn.get('total_vp_spent', 0) for turn in player.turn_stats)
            total_room_tiles = sum(turn.get('room_tiles_acquired', 0) for turn in player.turn_stats)

            data[f'p{position}_total_cards_played'] = total_cards_played
            data[f'p{position}_total_cards_discarded'] = total_cards_discarded
            data[f'p{position}_total_vp_earned'] = total_vp_earned
            data[f'p{position}_total_vp_spent'] = total_vp_spent
            data[f'p{position}_total_room_tiles_acquired'] = total_room_tiles

            works_by_type = {}
            for turn in player.turn_stats:
                for art_type, count in turn.get('works_by_type', {}).items():
                    works_by_type[art_type] = works_by_type.get(art_type, 0) + count

            for art_type in ['Crafts', 'Painting', 'Sculpture', 'Relic']:
                data[f'p{position}_works_{art_type.lower()}'] = works_by_type.get(art_type, 0)

            works_by_theme = {}
            for turn in player.turn_stats:
                for theme, count in turn.get('works_by_theme', {}).items():
                    works_by_theme[theme] = works_by_theme.get(theme, 0) + count

            for theme in ['Nature', 'Mythology', 'Society', 'Orientalism', 'No theme']:
                theme_key = theme.lower().replace(' ', '_')
                data[f'p{position}_works_{theme_key}'] = works_by_theme.get(theme, 0)

        winner = max(self.players, key=lambda p: (p.score, -sum(1 for slot in self.get_board(f"{p.name}_board").slots if not slot.is_empty())))
        winner_position = self.players.index(winner) + 1
        data['winner_position'] = winner_position
        data['winner_strategy'] = winner.ai_strategy.name if winner.ai_strategy else 'None'
        data['winner_score'] = winner.score

        scores = [p.score for p in self.players]
        data['max_score'] = max(scores)
        data['min_score'] = min(scores)
        data['score_difference'] = max(scores) - min(scores)
        data['avg_score'] = sum(scores) / len(scores)

        return data

    def setup_game(self, num_players: int = 2, strategy_classes: Optional[List] = None):
        """Set up advanced game mode."""
        self.log("\nPlayer Strategies:")
        player_names = ["player 1", "player 2", "player 3", "player 4"]
        for i in range(num_players):
            if strategy_classes and i < len(strategy_classes):
                strategy = strategy_classes[i]()
            else:
                strategy = get_random_strategy()
            player = ModernismeAdvancedPlayer(player_names[i], strategy=strategy)
            self.add_player(player)
            self.log(f"  {player_names[i]}: {strategy.name}")

        # Moda cards
        moda_tema_cards = self._create_moda_tema_cards()
        moda_conjunto_cards = self._create_moda_conjunto_cards()
        self.moda_tema = random.choice(moda_tema_cards)
        self.moda_conjunto = random.choice(moda_conjunto_cards)

        self.log(f"\nPublic Objectives:")
        self.log(f"  Theme Fashion: {self.moda_tema.name}")
        self.log(f"  Set Fashion: {self.moda_conjunto.name}")

        # Work deck
        works = self._create_work_deck()
        work_deck = Deck(works)
        work_deck.shuffle()
        self.add_deck("works", work_deck)

        # Artist deck
        artists = self._create_artist_deck()
        artist_deck = Deck(artists)
        artist_deck.shuffle()
        self.add_deck("artists", artist_deck)

        first_artist = artist_deck.draw(1)[0]
        self.artist_discard.append(first_artist)

        # Create advantage deck
        advantages = self._create_advantage_cards()
        random.shuffle(advantages)
        self.advantage_deck = advantages
        self.available_advantages = [self.advantage_deck.pop() for _ in range(4)]

        # Create room tiles
        room_tiles = self._create_room_tiles()
        random.shuffle(room_tiles)
        self.room_tile_bag = room_tiles
        self.available_room_tiles = [self.room_tile_bag.pop() for _ in range(5)]

        # Player boards
        for player in self.players:
            board = self._create_player_board(player.name)
            self.add_board(f"{player.name}_board", board)

        # Deal initial cards - 5 in advanced mode
        for player in self.players:
            work_deck.draw_to_hand(player.hand, 5)
            player.active_artists = artist_deck.draw(2)

        # Commission cards
        encargo_cards = self._create_encargo_cards()
        random.shuffle(encargo_cards)
        self.log(f"\nPlayers selecting secret commissions:")
        for player in self.players:
            options = encargo_cards[:3]
            encargo_cards = encargo_cards[3:]
            player.commission_card = random.choice(options)
            self.log(f"  {player.name} selected a secret commission")

        # Advanced mode: Each player picks an advantage card
        self.log(f"\nPlayers selecting advantage cards:")
        for player in self.players:
            if self.available_advantages:
                self.log(f"  {player.name}'s turn - Advantage Card Market:")
                for adv in self.available_advantages:
                    adv_type = adv.get_property("advantage_type")
                    description = adv.get_property("description", "")
                    self.log(f"    - {adv.name}: {description}")

                # Use strategy to select card
                if player.ai_strategy:
                    card = player.ai_strategy.select_advantage_card(player, self, self.available_advantages)
                else:
                    card = random.choice(self.available_advantages)

                self.available_advantages.remove(card)
                player.advantage_cards.append(card)
                self.log(f"  → Selected: {card.name}")

                # Replenish
                if self.advantage_deck:
                    new_card = self.advantage_deck.pop()
                    self.available_advantages.append(new_card)
                    self.log(f"  Market replenished with {new_card.name}")

        # Advanced mode: Each player picks a room tile
        self.log(f"\nPlayers selecting initial room tiles:")
        for player in self.players:
            if self.available_room_tiles:
                self.log(f"  {player.name}'s turn - Room Tile Market:")
                for tile in self.available_room_tiles:
                    tile_type = tile.get_property("tile_type")
                    is_theme = tile.get_property("is_theme_tile", False)
                    tile_category = "Theme" if is_theme else "Type"
                    self.log(f"    - {tile.name} ({tile_category}: {tile_type.value})")

                # Use strategy to select tile
                if player.ai_strategy:
                    tile = player.ai_strategy.select_room_tile(player, self, self.available_room_tiles)
                else:
                    tile = random.choice(self.available_room_tiles)

                self.available_room_tiles.remove(tile)

                # Use strategy to select which room to assign tile to
                all_rooms = ["Room 1 (3 slots)", "Room 2 (4 slots)", "Room 3 (3 slots)",
                           "Room 4 (2 slots)", "Room 5 (3 slots)"]
                available_rooms_list = [r for r in all_rooms if r not in player.room_tiles]

                if player.ai_strategy:
                    room = player.ai_strategy.select_room_for_tile(player, self, tile, available_rooms_list)
                    if not room:  # Strategy returned None, pick randomly
                        room = random.choice(available_rooms_list)
                else:
                    room = random.choice(available_rooms_list)

                player.room_tiles[room] = tile

                tile_type = tile.get_property("tile_type")
                is_theme = tile.get_property("is_theme_tile", False)
                tile_category = "Theme" if is_theme else "Type"
                self.log(f"  → Selected: {tile.name} ({tile_category}: {tile_type.value}) for {room}")

                # Replenish
                if self.room_tile_bag:
                    new_tile = self.room_tile_bag.pop()
                    self.available_room_tiles.append(new_tile)
                    self.log(f"  Market replenished with {new_tile.name}")

        self.log(f"\nGame setup complete with {num_players} players!")
        self.log(f"Advanced mode: 5 work cards, room tiles, advantage cards")

    def _create_work_deck(self) -> List[Card]:
        """Create the deck of 112 work cards."""
        works = []
        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        # Relics: 16 cards, all 5 VP
        for i in range(16):
            card = Card(f"Relic {i+1}", art_type=ArtType.RELIC, theme=Theme.NONE, vp=5)
            works.append(card)

        # Regular works
        for art_type in [ArtType.CRAFTS, ArtType.PAINTING, ArtType.SCULPTURE]:
            for theme in themes:
                for vp in [1, 2, 3, 4]:
                    for copy in range(2):
                        card = Card(
                            f"{art_type.value} {theme.value} {vp}VP-{copy+1}",
                            art_type=art_type, theme=theme, vp=vp
                        )
                        works.append(card)

        return works

    def _create_artist_deck(self) -> List[Card]:
        """Create the deck of 28 artist cards."""
        artists = []
        main_types = [
            (ArtType.CRAFTS, 8),
            (ArtType.PAINTING, 8),
            (ArtType.SCULPTURE, 8),
            (ArtType.RELIC, 4)
        ]
        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        for art_type, count in main_types:
            if art_type == ArtType.RELIC:
                for i in range(count):
                    card = Card(f"Antiquarian {i+1}", art_type=art_type, theme=Theme.NONE)
                    artists.append(card)
            else:
                cards_per_theme = count // 4
                type_name_map = {
                    ArtType.CRAFTS: "Craftsman",
                    ArtType.PAINTING: "Painter",
                    ArtType.SCULPTURE: "Sculptor"
                }
                for theme in themes:
                    for i in range(cards_per_theme):
                        artist_name = f"{type_name_map[art_type]} {theme.value} {i+1}"
                        card = Card(artist_name, art_type=art_type, theme=theme)
                        artists.append(card)

        return artists

    def _create_moda_tema_cards(self) -> List[Card]:
        """Create fashion theme cards."""
        cards = []
        for theme in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]:
            card = Card(
                f"3 {theme.value} works",
                objective_type="theme",
                required_theme=theme,
                required_count=3,
                vp=3
            )
            cards.append(card)
        return cards

    def _create_moda_conjunto_cards(self) -> List[Card]:
        """Create fashion set cards."""
        cards = []
        theme_combos = [
            ([Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY], "Green-Blue-Yellow adjacents"),
            ([Theme.NATURE, Theme.MYTHOLOGY, Theme.ORIENTALISM], "Green-Blue-Red adjacents"),
            ([Theme.NATURE, Theme.SOCIETY, Theme.ORIENTALISM], "Green-Yellow-Red adjacents"),
            ([Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM], "Blue-Yellow-Red adjacents"),
        ]
        for themes, name in theme_combos:
            card = Card(name, objective_type="adjacent_set", required_themes=themes, vp=3)
            cards.append(card)
        return cards

    def _create_encargo_cards(self) -> List[Card]:
        """Create commission cards."""
        cards = []
        combinations = [
            ("2 Crafts + 1 Painting", [ArtType.CRAFTS, ArtType.PAINTING], [2, 1], 3),
            ("2 Crafts + 1 Sculpture", [ArtType.CRAFTS, ArtType.SCULPTURE], [2, 1], 3),
            ("2 Paintings + 1 Crafts", [ArtType.PAINTING, ArtType.CRAFTS], [2, 1], 3),
            ("2 Paintings + 1 Sculpture", [ArtType.PAINTING, ArtType.SCULPTURE], [2, 1], 3),
            ("2 Sculptures + 1 Crafts", [ArtType.SCULPTURE, ArtType.CRAFTS], [2, 1], 3),
            ("2 Sculptures + 1 Painting", [ArtType.SCULPTURE, ArtType.PAINTING], [2, 1], 3),
        ]

        for name, req_types, req_counts, vp in combinations:
            for i in range(3):
                card = Card(
                    f"{name} #{i+1}",
                    objective_type="mixed",
                    required_types=req_types,
                    required_counts=req_counts,
                    vp=vp
                )
                cards.append(card)

        for i in range(2):
            card = Card(
                f"2 Relics #{i+1}",
                objective_type="type_count",
                required_type=ArtType.RELIC,
                required_count=2,
                vp=3
            )
            cards.append(card)

        return cards

    def _create_room_tiles(self) -> List[Card]:
        """Create 21 room tiles with types/themes."""
        tiles = []

        # Create tiles for each type (4 types x 3 = 12 tiles)
        for tile_type in [RoomTileType.CRAFTS, RoomTileType.PAINTING,
                         RoomTileType.SCULPTURE, RoomTileType.RELIC]:
            for i in range(3):
                tile = Card(
                    f"Room Tile {tile_type.value} {i+1}",
                    tile_type=tile_type,
                    is_theme_tile=False
                )
                tiles.append(tile)

        # Create tiles for each theme (4 themes x 2 = 8 tiles) + 1 extra
        theme_counts = {
            RoomTileType.NATURE: 2,
            RoomTileType.MYTHOLOGY: 2,
            RoomTileType.SOCIETY: 3,  # One extra
            RoomTileType.ORIENTALISM: 2
        }

        for tile_type, count in theme_counts.items():
            for i in range(count):
                tile = Card(
                    f"Room Tile {tile_type.value} {i+1}",
                    tile_type=tile_type,
                    is_theme_tile=True
                )
                tiles.append(tile)

        return tiles

    def _create_advantage_cards(self) -> List[Card]:
        """Create advantage cards."""
        cards = []

        advantage_types = [
            (AdvantageCardType.UNIVERSAL_EXHIBITION, "Redraw hand once", 2),
            (AdvantageCardType.PATRONAGE, "Change both active artists", 2),
            (AdvantageCardType.CHURCH_VISIT, "Rearrange artist discard pile", 2),
            (AdvantageCardType.IDEA_EXCHANGE, "Trade card with opponent", 2),
            (AdvantageCardType.ESPIONAGE, "Take from opponent discard", 2),
            (AdvantageCardType.PLAN_CHANGE, "Return placed work to hand", 2),
            (AdvantageCardType.REMODELING, "Move two placed works", 2),
            (AdvantageCardType.REFORM, "Refresh room tiles", 2),
        ]

        for adv_type, description, count in advantage_types:
            for i in range(count):
                card = Card(
                    f"{adv_type.value}",
                    advantage_type=adv_type,
                    description=description
                )
                cards.append(card)

        return cards

    def _create_player_board(self, player_name: str) -> Board:
        """Create a player board."""
        board = Board(f"{player_name}'s House")

        room_layout = [
            (0, "Room 1 (3 slots)", 1), (1, "Room 1 (3 slots)", 2), (2, "Room 1 (3 slots)", 3),
            (3, "Room 2 (4 slots)", 1), (4, "Room 2 (4 slots)", 2),
            (5, "Room 3 (3 slots)", 1), (6, "Room 3 (3 slots)", 2), (7, "Room 3 (3 slots)", 3),
            (8, "Room 2 (4 slots)", 3), (9, "Room 2 (4 slots)", 4),
            (10, "Room 4 (2 slots)", 1), (11, "Room 4 (2 slots)", 2),
            (12, "Room 5 (3 slots)", 1), (13, "Room 5 (3 slots)", 2), (14, "Room 5 (3 slots)", 3)
        ]

        slots_by_id = {}
        for space_id, room_name, room_slot_num in room_layout:
            slot = Slot(f"{room_name}_{room_slot_num}", max_cards=1)
            slot.set_property("room", room_name)
            slot.set_property("space_id", space_id)
            slot.set_property("adjacent_spaces", [])
            board.add_slot(slot)
            slots_by_id[space_id] = slot

        door_connections = [
            (0, 5), (2, 3), (7, 12), (11, 12), (8, 13)
        ]

        for space1, space2 in door_connections:
            slots_by_id[space1].get_property("adjacent_spaces").append(space2)
            slots_by_id[space2].get_property("adjacent_spaces").append(space1)

        return board

    def pick_room_tile_for_player(self, player: ModernismeAdvancedPlayer, room_name: str) -> Optional[Card]:
        """Pick a room tile for a player."""
        # Log available room tiles
        if self.available_room_tiles:
            self.log(f"    Room Tile Market:")
            for tile in self.available_room_tiles:
                tile_type = tile.get_property("tile_type")
                is_theme = tile.get_property("is_theme_tile", False)
                tile_category = "Theme" if is_theme else "Type"
                self.log(f"      - {tile.name} ({tile_category}: {tile_type.value})")

            # Use strategy to select tile
            if player.ai_strategy:
                tile = player.ai_strategy.select_room_tile(player, self, self.available_room_tiles)
            else:
                tile = random.choice(self.available_room_tiles)

            self.available_room_tiles.remove(tile)

            tile_type = tile.get_property("tile_type")
            is_theme = tile.get_property("is_theme_tile", False)
            tile_category = "Theme" if is_theme else "Type"
            self.log(f"    Selected: {tile.name} ({tile_category}: {tile_type.value}) for {room_name}")

            # Replenish
            if self.room_tile_bag:
                new_tile = self.room_tile_bag.pop()
                self.available_room_tiles.append(new_tile)
                self.log(f"    Market replenished with {new_tile.name}")

            return tile
        elif self.room_tile_bag:
            tile = self.room_tile_bag.pop()
            tile_type = tile.get_property("tile_type")
            is_theme = tile.get_property("is_theme_tile", False)
            tile_category = "Theme" if is_theme else "Type"
            self.log(f"    Selected from bag: {tile.name} ({tile_category}: {tile_type.value}) for {room_name}")
            return tile
        return None

    def use_advantage_card(self, player: ModernismeAdvancedPlayer, card: Card, phase: str) -> bool:
        """
        Use an advantage card and apply its effect.

        Args:
            player: The player using the card
            card: The advantage card to use
            phase: Current game phase

        Returns:
            True if card was successfully used, False otherwise
        """
        card_type = card.get_property("advantage_type")
        self.log(f"    {player.name} uses advantage card: {card.name}")

        if card_type == AdvantageCardType.UNIVERSAL_EXHIBITION:
            # Redraw hand once
            work_deck = self.get_deck("works")
            cards_to_return = len(player.hand.cards)
            returned = []
            for _ in range(cards_to_return):
                if player.hand.cards:
                    c = player.hand.cards[0]
                    player.hand.remove_card(c)
                    returned.append(c)

            # Shuffle returned cards back
            work_deck.cards.extend(returned)
            work_deck.shuffle()

            # Draw new hand
            work_deck.draw_to_hand(player.hand, cards_to_return)
            self.log(f"      → Redrew {cards_to_return} cards")
            return True

        elif card_type == AdvantageCardType.PATRONAGE:
            # Change both active artists
            artist_deck = self.get_deck("artists")
            if len(artist_deck.cards) >= 2:
                old_artists = player.active_artists[:]
                player.active_artists = artist_deck.draw(2)
                self.artist_discard.extend(old_artists)
                self.log(f"      → Replaced both artists: {[a.name for a in player.active_artists]}")
                return True
            return False

        elif card_type == AdvantageCardType.CHURCH_VISIT:
            # Rearrange artist discard pile - for AI, just shuffle
            random.shuffle(self.artist_discard)
            self.log(f"      → Rearranged artist discard pile")
            return True

        elif card_type == AdvantageCardType.IDEA_EXCHANGE:
            # Trade card with opponent - simplified: swap a random card
            if len(self.players) > 1:
                opponent = random.choice([p for p in self.players if p != player])
                if player.hand.cards and opponent.hand.cards:
                    player_card = random.choice(player.hand.cards)
                    opponent_card = random.choice(opponent.hand.cards)
                    player.hand.remove_card(player_card)
                    opponent.hand.remove_card(opponent_card)
                    player.hand.add_card(opponent_card)
                    opponent.hand.add_card(player_card)
                    self.log(f"      → Swapped cards with {opponent.name}")
                    return True
            return False

        elif card_type == AdvantageCardType.ESPIONAGE:
            # Take from opponent's discard
            if len(self.players) > 1:
                opponents_with_discards = [p for p in self.players if p != player and p.discard_pile]
                if opponents_with_discards:
                    opponent = random.choice(opponents_with_discards)
                    stolen_card = opponent.discard_pile.pop()
                    player.hand.add_card(stolen_card)
                    self.log(f"      → Took {stolen_card.name} from {opponent.name}'s discard")
                    return True
            return False

        elif card_type == AdvantageCardType.PLAN_CHANGE:
            # Return placed work to hand
            board = self.get_board(f"{player.name}_board")
            if board:
                occupied_slots = [s for s in board.slots if not s.is_empty()]
                if occupied_slots:
                    slot = random.choice(occupied_slots)
                    work = slot.get_cards()[0]
                    slot.remove_card(work)
                    player.hand.add_card(work)
                    # Deduct the VP that was earned
                    work_vp = work.get_property("vp", 0)
                    player.add_score(-work_vp)
                    self.log(f"      → Returned {work.name} to hand, deducted {work_vp} VP")
                    return True
            return False

        elif card_type == AdvantageCardType.REMODELING:
            # Move two placed works - simplified: just log for now
            self.log(f"      → Remodeling (effect not fully implemented)")
            return True

        elif card_type == AdvantageCardType.REFORM:
            # Refresh room tiles
            if len(self.available_room_tiles) > 0 and len(self.room_tile_bag) > 0:
                # Return all available tiles to bag
                self.room_tile_bag.extend(self.available_room_tiles)
                random.shuffle(self.room_tile_bag)
                # Draw new tiles
                self.available_room_tiles = [self.room_tile_bag.pop() for _ in range(min(5, len(self.room_tile_bag)))]
                self.log(f"      → Refreshed room tile market")
                return True
            return False

        return False

    def handle_milestone_reward(self, player: ModernismeAdvancedPlayer) -> None:
        """Handle giving advantage card when player reaches milestone."""
        if self.available_advantages:
            self.log(f"      Advantage Card Market:")
            for adv in self.available_advantages:
                adv_type = adv.get_property("advantage_type")
                description = adv.get_property("description", "")
                self.log(f"        - {adv.name}: {description}")

            # Use strategy to select card
            if player.ai_strategy:
                card = player.ai_strategy.select_advantage_card(player, self, self.available_advantages)
            else:
                card = random.choice(self.available_advantages)

            self.available_advantages.remove(card)
            player.advantage_cards.append(card)
            self.log(f"      → Selected advantage card: {card.name}")

            # Replenish
            if self.advantage_deck:
                new_card = self.advantage_deck.pop()
                self.available_advantages.append(new_card)
                self.log(f"      Market replenished with {new_card.name}")

    def play_talent_hunt_phase(self, player: ModernismeAdvancedPlayer, neighbor: Optional[ModernismeAdvancedPlayer] = None) -> None:
        """Execute the talent hunt phase with neighbor interaction."""
        artist_deck = self.get_deck("artists")

        # Reshuffle if needed
        if artist_deck.is_empty():
            artist_deck.cards = self.artist_discard[:-1]
            artist_deck.shuffle()
            self.artist_discard = [self.artist_discard[-1]]

        # Draw 2 artists for player to choose from
        drawn_artists = artist_deck.draw(min(2, len(artist_deck.cards)))
        if len(drawn_artists) < 2 and not artist_deck.is_empty():
            # If we only got 1, try to draw another after reshuffling
            if artist_deck.is_empty():
                artist_deck.cards = self.artist_discard[:-1]
                artist_deck.shuffle()
                self.artist_discard = [self.artist_discard[-1]]
            if not artist_deck.is_empty():
                drawn_artists.extend(artist_deck.draw(1))

        if not drawn_artists:
            self.log(f"  No artists available to hire!")
            return

        self.log(f"  Available artists: {[a.name for a in drawn_artists]}")

        # Player selects which artist to hire
        if player.ai_strategy and len(drawn_artists) > 1:
            selected_artist = player.ai_strategy.select_artist_to_hire(player, self, drawn_artists)
        else:
            selected_artist = drawn_artists[0]

        # Player selects which of their active artists to dismiss
        if player.ai_strategy:
            artist_to_dismiss = player.ai_strategy.select_artist_to_dismiss(player, self, player.active_artists, selected_artist)
        else:
            artist_to_dismiss = player.active_artists[0]

        # Replace the dismissed artist with the new one
        dismiss_index = player.active_artists.index(artist_to_dismiss)
        player.active_artists[dismiss_index] = selected_artist

        self.log(f"  {player.name} hired {selected_artist.name}, dismissed {artist_to_dismiss.name}")

        # Prepare options for neighbor to steal
        unchosen_artists = [a for a in drawn_artists if a != selected_artist]
        available_for_neighbor = unchosen_artists + [artist_to_dismiss]

        # Neighbor can steal one of the unchosen or dismissed artists
        if neighbor and available_for_neighbor:
            self.log(f"  {neighbor.name} can steal from: {[a.name for a in available_for_neighbor]}")

            if neighbor.ai_strategy:
                stolen_artist = neighbor.ai_strategy.steal_artist_from_neighbor(neighbor, self, available_for_neighbor)
            else:
                # Random chance to steal (30% chance)
                stolen_artist = random.choice(available_for_neighbor) if random.random() < 0.3 else None

            if stolen_artist:
                # Neighbor steals the artist
                available_for_neighbor.remove(stolen_artist)

                # Neighbor must dismiss one of their artists
                if neighbor.ai_strategy:
                    neighbor_dismisses = neighbor.ai_strategy.select_artist_to_dismiss(neighbor, self, neighbor.active_artists, stolen_artist)
                else:
                    neighbor_dismisses = neighbor.active_artists[0]

                dismiss_index = neighbor.active_artists.index(neighbor_dismisses)
                neighbor.active_artists[dismiss_index] = stolen_artist

                self.log(f"  → {neighbor.name} stole {stolen_artist.name}, dismissed {neighbor_dismisses.name}")

                # Neighbor's dismissed artist goes to discard
                self.artist_discard.append(neighbor_dismisses)

        # All remaining artists go to discard
        for artist in available_for_neighbor:
            self.artist_discard.append(artist)
            self.log(f"  {artist.name} → artist discard pile")

    def end_season(self) -> None:
        """Handle end of season."""
        self.log(f"\n=== End of Season {self.season} ===")

        # Pass discards
        discards_to_pass = []
        for player in self.players:
            discards_to_pass.append(player.discard_pile[:])
            player.discard_pile = []

        self.log("\nPassing discards counterclockwise:")
        for i, player in enumerate(self.players):
            next_idx = (i + 1) % len(self.players)
            cards_received = discards_to_pass[i]
            if cards_received:
                self.log(f"  {self.players[next_idx].name} receives {len(cards_received)} cards from {player.name}")
                for card in cards_received:
                    self.players[next_idx].hand.add_card(card)

        # Refill hands - variable by season in advanced mode
        work_deck = self.get_deck("works")
        hand_sizes = {1: 5, 2: 5, 3: 6, 4: 7}  # Spring, Summer, Autumn, Winter
        target_size = hand_sizes.get(self.season + 1, 6)

        self.log(f"\nRefilling hands to {target_size} cards:")
        for player in self.players:
            cards_before = len(player.hand)
            while len(player.hand) < target_size and not work_deck.is_empty():
                cards = work_deck.draw(1)
                if cards:
                    player.hand.add_card(cards[0])
            cards_drawn = len(player.hand) - cards_before
            if cards_drawn > 0:
                self.log(f"  {player.name} drew {cards_drawn} cards (now has {len(player.hand)})")

        self.first_player_idx = (self.first_player_idx - 1) % len(self.players)
        self.log(f"\nFirst player for next season: {self.players[self.first_player_idx].name}")

        self.season += 1

    def _check_adjacent_set(self, player: ModernismeAdvancedPlayer, required_themes: List[Theme]) -> int:
        """Check how many times a player has completed an adjacent set."""
        board = self.get_board(f"{player.name}_board")

        works_by_space = {}
        for slot in board.slots:
            if not slot.is_empty():
                space_id = slot.get_property("space_id")
                work = slot.get_cards()[0]
                works_by_space[space_id] = work

        count = 0
        checked_sets = set()

        for space_id, work in works_by_space.items():
            slot = next(s for s in board.slots if s.get_property("space_id") == space_id)
            adjacent_spaces = slot.get_property("adjacent_spaces")

            for adj1 in adjacent_spaces:
                if adj1 not in works_by_space:
                    continue
                for adj2 in adjacent_spaces:
                    if adj2 not in works_by_space or adj2 == adj1:
                        continue

                    works = [work, works_by_space[adj1], works_by_space[adj2]]
                    themes = set()
                    for w in works:
                        if w.get_property("art_type") == ArtType.RELIC and w in player.reliquia_themes:
                            themes.add(player.reliquia_themes[w])
                        else:
                            themes.add(w.get_property("theme"))

                    if themes == set(required_themes):
                        set_id = tuple(sorted([space_id, adj1, adj2]))
                        if set_id not in checked_sets:
                            checked_sets.add(set_id)
                            count += 1

        return count

    def _score_objectives(self):
        """Score all objectives."""
        self.log("\n" + "=" * 60)
        self.log("OBJECTIVE SCORING - Moda & Encargo")
        self.log("=" * 60)

        for player in self.players:
            board = self.get_board(f"{player.name}_board")
            total_objective_vp = 0

            self.log(f"\n{player.name}:")

            works = []
            for slot in board.slots:
                if not slot.is_empty():
                    works.append(slot.get_cards()[0])

            type_counts = {}
            theme_counts = {}
            for work in works:
                art_type = work.get_property("art_type")
                if art_type == ArtType.RELIC and work in player.reliquia_themes:
                    theme = player.reliquia_themes[work]
                else:
                    theme = work.get_property("theme")
                type_counts[art_type] = type_counts.get(art_type, 0) + 1
                if theme != Theme.NONE:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1

            # Moda tema
            if self.moda_tema:
                required_theme = self.moda_tema.get_property("required_theme")
                required_count = self.moda_tema.get_property("required_count")
                vp_per = self.moda_tema.get_property("vp")

                actual_count = theme_counts.get(required_theme, 0)
                times_completed = actual_count // required_count
                if times_completed > 0:
                    vp_earned = times_completed * vp_per
                    total_objective_vp += vp_earned
                    self.log(f"  Theme Fashion ({self.moda_tema.name}): {times_completed}x = +{vp_earned} VP")

            # Moda conjunto
            if self.moda_conjunto:
                required_themes = self.moda_conjunto.get_property("required_themes")
                vp_per = self.moda_conjunto.get_property("vp")

                times_completed = self._check_adjacent_set(player, required_themes)
                if times_completed > 0:
                    vp_earned = times_completed * vp_per
                    total_objective_vp += vp_earned
                    self.log(f"  Set Fashion ({self.moda_conjunto.name}): {times_completed}x = +{vp_earned} VP")

            # Encargo
            if player.commission_card:
                encargo = player.commission_card
                obj_type = encargo.get_property("objective_type")
                vp_per = encargo.get_property("vp")
                times_completed = 0

                if obj_type == "type_count":
                    required_type = encargo.get_property("required_type")
                    required_count = encargo.get_property("required_count")
                    actual_count = type_counts.get(required_type, 0)
                    times_completed = actual_count // required_count

                elif obj_type == "theme_count":
                    required_theme = encargo.get_property("required_theme")
                    required_count = encargo.get_property("required_count")
                    actual_count = theme_counts.get(required_theme, 0)
                    times_completed = actual_count // required_count

                elif obj_type == "mixed":
                    required_types = encargo.get_property("required_types")
                    required_counts = encargo.get_property("required_counts")
                    min_sets = float('inf')
                    for req_type, req_count in zip(required_types, required_counts):
                        actual = type_counts.get(req_type, 0)
                        min_sets = min(min_sets, actual // req_count)
                    times_completed = int(min_sets) if min_sets != float('inf') else 0

                if times_completed > 0:
                    vp_earned = times_completed * vp_per
                    total_objective_vp += vp_earned
                    self.log(f"  Secret Commission ({encargo.name}): {times_completed}x = +{vp_earned} VP")
                else:
                    self.log(f"  Secret Commission ({encargo.name}): not completed")

            if total_objective_vp > 0:
                player.add_score(total_objective_vp)

    def _score_room_bonuses(self):
        """Score room completion bonuses with advanced rules."""
        self.log("\n" + "=" * 60)
        self.log("ROOM COMPLETION BONUSES (Advanced Mode)")
        self.log("=" * 60)

        for player in self.players:
            board = self.get_board(f"{player.name}_board")
            self.log(f"\n{player.name}:")

            rooms = {}
            for slot in board.slots:
                room_name = slot.get_property("room")
                if room_name not in rooms:
                    rooms[room_name] = []
                if not slot.is_empty():
                    rooms[room_name].append(slot.get_cards()[0])

            room_configs = {
                "Room 1 (3 slots)": 3,
                "Room 2 (4 slots)": 4,
                "Room 3 (3 slots)": 3,
                "Room 4 (2 slots)": 2,
                "Room 5 (3 slots)": 3
            }

            for room_name, required_works in room_configs.items():
                works_in_room = rooms.get(room_name, [])
                room_status = f"  {room_name}: {len(works_in_room)}/{required_works} works"

                if len(works_in_room) == required_works:
                    room_status += " (COMPLETE)"

                    # Check if player has room tile
                    if room_name not in player.room_tiles:
                        room_status += " - No room tile, no bonus"
                        self.log(room_status)
                        continue

                    room_tile = player.room_tiles[room_name]
                    tile_type = room_tile.get_property("tile_type")
                    is_theme_tile = room_tile.get_property("is_theme_tile", False)

                    # Check if all same type
                    types = [w.get_property("art_type") for w in works_in_room]
                    same_type = len(set(types)) == 1

                    # Check if all same theme
                    themes = []
                    for w in works_in_room:
                        if w.get_property("art_type") == ArtType.RELIC and w in player.reliquia_themes:
                            themes.append(player.reliquia_themes[w])
                        elif w.get_property("theme") != Theme.NONE:
                            themes.append(w.get_property("theme"))
                    same_theme = len(themes) == required_works and len(set(themes)) == 1

                    bonus = 0
                    bonus_type = ""

                    # Advanced mode: tile must match
                    if same_type and not is_theme_tile:
                        # Check if tile type matches work type
                        work_type = types[0]
                        if tile_type.value == work_type.value:
                            bonus = required_works
                            bonus_type = "same type"

                            # Check if all different themes for doubling
                            unique_themes = set()
                            for w in works_in_room:
                                if w.get_property("art_type") == ArtType.RELIC and w in player.reliquia_themes:
                                    unique_themes.add(player.reliquia_themes[w])
                                elif w.get_property("theme") != Theme.NONE:
                                    unique_themes.add(w.get_property("theme"))

                            if len(unique_themes) == required_works:
                                bonus *= 2
                                bonus_type += " (DOUBLED - all different themes)"

                    elif same_theme and is_theme_tile:
                        # Check if tile theme matches work theme
                        work_theme = themes[0]
                        if tile_type.value == work_theme.value:
                            bonus = required_works
                            bonus_type = "same theme"

                            # Check if all different types for doubling
                            unique_types = set(types)
                            if len(unique_types) == required_works:
                                bonus *= 2
                                bonus_type += " (DOUBLED - all different types)"

                    if bonus > 0:
                        player.add_score(bonus)
                        room_status += f" → {bonus_type}: +{bonus} VP"
                    else:
                        room_status += " - Room tile doesn't match bonus type"

                self.log(room_status)


def play_modernisme_advanced_game(log_file: Optional[TextIO] = None, num_players: int = 4,
                                   strategy_classes: Optional[List] = None):
    """Play a complete game of Modernisme in advanced mode."""
    game = ModernismeAdvancedGame(log_file=log_file)

    game.log("=" * 60)
    game.log("MODERNISME ADVANCED MODE")
    game.log("=" * 60)

    game.setup_game(num_players=num_players, strategy_classes=strategy_classes)

    turn_counter = 0
    for season in range(1, 5):
        game.log(f"\n{'='*60}")
        game.log(f"SEASON {season}")
        game.log(f"First player: {game.players[game.first_player_idx].name}")
        game.log('='*60)

        for offset in range(len(game.players)):
            player_idx = (game.first_player_idx + offset) % len(game.players)
            player = game.players[player_idx]

            turn_counter += 1
            player.start_turn(season, turn_counter)

            game.log(f"\n--- {player.name}'s Turn ---")

            game.log("Hand at start of turn:")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            game.log(f"  {', '.join(hand_cards)}")

            # Check for advantage card usage before talent hunt
            if player.advantage_cards and player.ai_strategy:
                card_to_use = player.ai_strategy.should_use_advantage_card(player, game, 'before_talent_hunt')
                if card_to_use and card_to_use in player.advantage_cards:
                    if game.use_advantage_card(player, card_to_use, 'before_talent_hunt'):
                        player.advantage_cards.remove(card_to_use)

            game.log("\nPhase 1: Talent Hunt")
            # Determine the next player (neighbor) for artist stealing
            next_player_idx = (player_idx + 1) % len(game.players)
            neighbor = game.players[next_player_idx]
            game.play_talent_hunt_phase(player, neighbor)

            # Check for advantage card usage before placement
            if player.advantage_cards and player.ai_strategy:
                card_to_use = player.ai_strategy.should_use_advantage_card(player, game, 'before_placement')
                if card_to_use and card_to_use in player.advantage_cards:
                    if game.use_advantage_card(player, card_to_use, 'before_placement'):
                        player.advantage_cards.remove(card_to_use)

            game.log("Phase 2: Placement (including room tile acquisition)")
            game.log(f"  Active artists: {[a.name for a in player.active_artists]}")
            game.log(f"  Room tiles owned: {list(player.room_tiles.keys())}")
            game.log(f"  Advantage cards held: {len(player.advantage_cards)}")
            player.strategy(game)

            # Check for advantage card usage after placement
            if player.advantage_cards and player.ai_strategy:
                card_to_use = player.ai_strategy.should_use_advantage_card(player, game, 'after_placement')
                if card_to_use and card_to_use in player.advantage_cards:
                    if game.use_advantage_card(player, card_to_use, 'after_placement'):
                        player.advantage_cards.remove(card_to_use)

            game.log(f"\nHand at end of turn ({len(player.hand)} cards):")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            if hand_cards:
                game.log(f"  {', '.join(hand_cards)}")
            else:
                game.log(f"  (empty)")
            game.log(f"Score: {player.score} VP")

            player.end_turn()

        if season < 4:
            game.end_season()

    # Contextualize Relics
    game.log("\n" + "=" * 60)
    game.log("CONTEXTUALIZING RELICS")
    game.log("=" * 60)
    for player in game.players:
        player.assign_reliquia_themes(game)

    # Score objectives
    game._score_objectives()

    # Score room bonuses
    game._score_room_bonuses()

    # Determine winner
    game.log("\n" + "=" * 60)
    game.log("FINAL SCORES")
    game.log("=" * 60)

    for player in game.players:
        board = game.get_board(f"{player.name}_board")
        total_works = sum(1 for slot in board.slots if not slot.is_empty())
        game.log(f"{player.name}: {player.score} VP ({total_works} works placed, {len(player.room_tiles)} room tiles)")

    winner = max(game.players, key=lambda p: (p.score, -sum(1 for slot in game.get_board(f"{p.name}_board").slots if not slot.is_empty())))

    game.log("\n" + "=" * 60)
    game.log(f"🏆 WINNER: {winner.name} with {winner.score} VP!")
    game.log("=" * 60)

    return game


if __name__ == "__main__":
    play_modernisme_advanced_game()
