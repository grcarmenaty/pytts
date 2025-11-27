"""
Modernisme - A complete implementation of the tabletop game.

This example implements the full Modernisme game based on the rules v0.4.1,
set in Barcelona's Modernist movement of the 19th century.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Game, Player, Card, Deck, Board, Slot, Hand, DiscardPile
from pytts.strategy import Strategy, get_random_strategy
import random
from typing import List, Optional, Tuple, TextIO
from enum import Enum
from io import StringIO


class ArtType(Enum):
    """Types of art/artists in the game."""
    CRAFTS = "Crafts"  # Craftsman
    PAINTING = "Painting"  # Painter
    SCULPTURE = "Sculpture"  # Sculptor
    RELIC = "Relic"  # Antiquarian


class Theme(Enum):
    """Themes for works and artists."""
    NATURE = "Nature"  # Green
    MYTHOLOGY = "Mythology"  # Blue
    SOCIETY = "Society"  # Yellow
    ORIENTALISM = "Orientalism"  # Red
    NONE = "No theme"  # For relics


class ModernismePlayer(Player):
    """Player in Modernisme game."""

    def __init__(self, name: str, strategy: Optional[Strategy] = None):
        super().__init__(name, hand_max_size=None)  # No hand limit
        self.active_artists: List[Card] = []  # 2 active artist cards
        self.discard_pile: List[Card] = []  # Player's work card discards
        self.commission_card: Optional[Card] = None  # Secret objective
        self.completed_rooms: List[int] = []  # Track which rooms are complete
        self.ai_strategy: Optional[Strategy] = strategy  # AI strategy
        self.reliquia_themes: dict = {}  # Maps reliquia card to assigned theme

    def strategy(self, game: 'ModernismeGame') -> None:
        """
        AI strategy for Modernisme.

        Uses the assigned AI strategy to select which works to commission.
        Commissions 1-2 works per turn based on strategy recommendations.
        """
        # Commission 1-2 works per turn
        works_to_commission = min(2, len(self.hand.cards))

        for _ in range(works_to_commission):
            if not self.hand.cards:
                break

            # Use AI strategy to select a work
            if self.ai_strategy:
                work = self.ai_strategy.select_work(self, game)
            else:
                # Fallback: select first commissionable work
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
                continue  # Successfully commissioned
            else:
                break  # Can't commission anymore

        # Reset pool at end of turn
        if self.pool > 0:
            game.log(f"  {self.pool} VP remaining in pool discarded at end of turn")
        self.pool = 0

    def _try_commission_work(self, work: Card, game: 'ModernismeGame') -> bool:
        """Try to commission a specific work using the VP pool system."""
        work_vp = work.get_property("vp", 0)

        # Check if we need to discard more cards to reach required VP
        vp_needed = work_vp - self.pool

        if vp_needed > 0:
            # Need to discard more cards
            available_cards = [c for c in self.hand.cards if c != work]

            if not available_cards:
                return False

            # Find the best combination of cards to discard
            card_vps = [(card, card.get_property("vp", 0)) for card in available_cards]
            card_vps.sort(key=lambda x: x[1])

            best_discard = None
            best_overspend = float('inf')

            # Try to find combinations that meet the requirement
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

            # Also try greedy approach from smallest to largest
            discard_cards = []
            total_vp = 0
            for card, vp in card_vps:
                if total_vp < vp_needed:
                    discard_cards.append(card)
                    total_vp += vp

            if total_vp >= vp_needed:
                overspend = total_vp - vp_needed
                if overspend < best_overspend:
                    best_overspend = overspend
                    best_discard = discard_cards[:]

            if best_discard is None:
                return False

            # Discard cards and add their VP to the pool
            discard_vp = sum(card.get_property("vp", 0) for card in best_discard)
            discard_names = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in best_discard]
            game.log(f"    Discarded: {', '.join(discard_names)} (added {discard_vp} VP to pool)")

            for card in best_discard:
                self.hand.remove_card(card)
                self.discard_pile.append(card)
                self.pool += card.get_property("vp", 0)

        # Now we should have enough VP in the pool
        if self.pool < work_vp:
            return False

        game.log(f"    Pool before commission: {self.pool} VP")

        # Spend from pool
        self.pool -= work_vp
        game.log(f"    Spent {work_vp} VP from pool, remaining: {self.pool} VP")

        # Remove work from hand
        self.hand.remove_card(work)

        # Find first available room slot
        board = game.get_board(f"{self.name}_board")
        if board:
            for slot in board.get_empty_slots():
                if slot.can_place_card(work, self):
                    slot.place_card(work, self)

                    # Score immediate VP
                    base_vp = work_vp
                    bonus_vp = 0

                    # Bonus VP if artist theme matches work theme
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

                    # Detailed logging
                    room_name = slot.get_property("room")
                    space_id = slot.get_property("space_id")
                    game.log(f"    → Placed '{work.name}' in {room_name} (space {space_id})")
                    bonus_msg = f" + {bonus_vp} (theme bonus from {matching_artist.name})" if bonus_vp > 0 else ""
                    game.log(f"      Base VP: {base_vp}{bonus_msg} = {vp_gain} VP total")
                    game.log(f"      {self.name}'s total score: {self.score} VP")

                    return True

        return False

    def assign_reliquia_themes(self, game: 'ModernismeGame') -> None:
        """
        Assign optimal themes to all placed Reliquias.

        This is done at the end of the game to maximize bonuses.
        """
        board = game.get_board(f"{self.name}_board")
        if not board:
            return

        # Find all placed Reliquias
        reliquias = []
        for slot in board.slots:
            if not slot.is_empty():
                work = slot.get_cards()[0]
                if work.get_property("art_type") == ArtType.RELIC:
                    reliquias.append((work, slot))

        if not reliquias:
            return

        game.log(f"\n{self.name} contextualizing Relics:")

        # For each Relic, determine the best theme
        for work, slot in reliquias:
            best_theme = self._find_best_theme_for_reliquia(work, slot, game)
            self.reliquia_themes[work] = best_theme
            game.log(f"  {work.name} → {best_theme.value} theme")

    def _find_best_theme_for_reliquia(self, reliquia: Card, slot: 'Slot', game: 'ModernismeGame') -> Theme:
        """Find the best theme to assign to a Reliquia."""
        board = game.get_board(f"{self.name}_board")
        room_name = slot.get_property("room")

        # Score each theme option
        theme_scores = {theme: 0 for theme in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}

        # Check which theme would complete room bonus
        for theme in theme_scores.keys():
            # Temporarily assign this theme
            temp_theme_count = {t: 0 for t in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}
            room_works = []

            for s in board.slots:
                if s.get_property("room") == room_name and not s.is_empty():
                    w = s.get_cards()[0]
                    room_works.append(w)
                    if w == reliquia:
                        temp_theme_count[theme] += 1
                    elif w.get_property("art_type") == ArtType.RELIC and w in self.reliquia_themes:
                        temp_theme_count[self.reliquia_themes[w]] += 1
                    elif w.get_property("theme") != Theme.NONE:
                        temp_theme_count[w.get_property("theme")] += 1

            # Check if this would give same-theme bonus
            room_size = sum(1 for s in board.slots if s.get_property("room") == room_name)
            if len(room_works) == room_size:
                # Room is complete
                for t, count in temp_theme_count.items():
                    if count == room_size and t == theme:
                        theme_scores[theme] += room_size * 2  # Room bonus worth a lot

        # Check which theme helps with moda_tema
        if game.moda_tema:
            required_theme = game.moda_tema.get_property("required_theme")
            if required_theme in theme_scores:
                theme_scores[required_theme] += 3  # Moda tema worth 3 VP

        # Default: choose theme that appears most in player's works (for moda conjunto potential)
        theme_counts = {t: 0 for t in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]}
        for s in board.slots:
            if not s.is_empty():
                w = s.get_cards()[0]
                if w != reliquia:
                    if w.get_property("art_type") == ArtType.RELIC and w in self.reliquia_themes:
                        theme_counts[self.reliquia_themes[w]] += 1
                    elif w.get_property("theme") != Theme.NONE:
                        theme_counts[w.get_property("theme")] += 1

        # Add small bonus for balancing themes (for moda conjunto)
        min_count = min(theme_counts.values())
        for theme in theme_scores.keys():
            if theme_counts[theme] == min_count:
                theme_scores[theme] += 1  # Small bonus for less common themes

        # Return theme with highest score
        best_theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        return best_theme


class ModernismeGame(Game):
    """The Modernisme game implementation."""

    def __init__(self, log_file: Optional[TextIO] = None):
        super().__init__("Modernisme")
        self.artist_discard: List[Card] = []  # Artist discard pile
        self.season = 1  # Current season (1-4)
        self.first_player_idx = 0  # Index of first player
        self.moda_tema: Optional[Card] = None  # Public theme fashion card
        self.moda_conjunto: Optional[Card] = None  # Public set fashion card
        self.log_file = log_file  # Optional file to log game output

    def log(self, message: str = ""):
        """Log a message to file or console."""
        if self.log_file:
            self.log_file.write(message + "\n")
        else:
            print(message)

    def get_game_data(self) -> dict:
        """
        Get structured game data for CSV export.

        Returns:
            Dictionary containing game results and player statistics
        """
        data = {
            'moda_tema': self.moda_tema.name if self.moda_tema else '',
            'moda_conjunto': self.moda_conjunto.name if self.moda_conjunto else '',
        }

        # Add player data
        for i, player in enumerate(self.players):
            position = i + 1  # 1-indexed position
            board = self.get_board(f"{player.name}_board")
            total_works = sum(1 for slot in board.slots if not slot.is_empty()) if board else 0

            data[f'player_{position}_name'] = player.name
            data[f'player_{position}_strategy'] = player.ai_strategy.name if player.ai_strategy else 'None'
            data[f'player_{position}_score'] = player.score
            data[f'player_{position}_works'] = total_works
            data[f'player_{position}_commission'] = player.commission_card.name if player.commission_card else ''

        # Determine winner
        winner = max(self.players, key=lambda p: (p.score, -sum(1 for slot in self.get_board(f"{p.name}_board").slots if not slot.is_empty())))
        winner_position = self.players.index(winner) + 1
        data['winner_position'] = winner_position
        data['winner_strategy'] = winner.ai_strategy.name if winner.ai_strategy else 'None'
        data['winner_score'] = winner.score

        # Calculate score statistics
        scores = [p.score for p in self.players]
        data['max_score'] = max(scores)
        data['min_score'] = min(scores)
        data['score_difference'] = max(scores) - min(scores)

        return data

    def setup_game(self, num_players: int = 2):
        """Set up a game of Modernisme."""
        # Create players with random strategies
        self.log("\nPlayer Strategies:")
        player_names = ["player 1", "player 2", "player 3", "player 4"]
        for i in range(num_players):
            strategy = get_random_strategy()
            player = ModernismePlayer(player_names[i], strategy=strategy)
            self.add_player(player)
            self.log(f"  {player_names[i]}: {strategy.name}")

        # Select moda cards (public objectives)
        moda_tema_cards = self._create_moda_tema_cards()
        moda_conjunto_cards = self._create_moda_conjunto_cards()
        self.moda_tema = random.choice(moda_tema_cards)
        self.moda_conjunto = random.choice(moda_conjunto_cards)

        self.log(f"\nPublic Objectives:")
        self.log(f"  Theme Fashion: {self.moda_tema.name}")
        self.log(f"  Set Fashion: {self.moda_conjunto.name}")

        # Create work deck
        works = self._create_work_deck()
        work_deck = Deck(works)
        work_deck.shuffle()
        self.add_deck("works", work_deck)

        # Create artist deck
        artists = self._create_artist_deck()
        artist_deck = Deck(artists)
        artist_deck.shuffle()
        self.add_deck("artists", artist_deck)

        # Initialize artist discard pile with one card
        first_artist = artist_deck.draw(1)[0]
        self.artist_discard.append(first_artist)

        # Create player boards
        for player in self.players:
            board = self._create_player_board(player.name)
            self.add_board(f"{player.name}_board", board)

        # Deal initial cards
        for player in self.players:
            # Draw 6 work cards
            work_deck.draw_to_hand(player.hand, 6)

            # Draw 2 artists as active artists
            player.active_artists = artist_deck.draw(2)

        # Each player selects a secret commission
        encargo_cards = self._create_encargo_cards()
        random.shuffle(encargo_cards)
        self.log(f"\nPlayers selecting secret commissions:")
        for player in self.players:
            # Draw 3, pick 1
            options = encargo_cards[:3]
            encargo_cards = encargo_cards[3:]  # Remove used cards
            # AI randomly picks one
            player.commission_card = random.choice(options)
            self.log(f"  {player.name} selected a secret commission")

        self.log(f"\nGame setup complete with {num_players} players!")
        self.log(f"Each player has 6 work cards, 2 active artists, and 1 secret commission.")

    def _create_work_deck(self) -> List[Card]:
        """Create the deck of 112 work cards matching xlsx specifications."""
        works = []

        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        # Relics: 16 cards, all 5 VP
        for i in range(16):
            card = Card(
                f"Relic {i+1}",
                art_type=ArtType.RELIC,
                theme=Theme.NONE,
                vp=5
            )
            works.append(card)

        # Regular works: 32 of each type (Crafts, Painting, Sculpture)
        # For each type: 2 cards of each (theme x VP) combination
        # VP values: 1, 2, 3, 4
        for art_type in [ArtType.CRAFTS, ArtType.PAINTING, ArtType.SCULPTURE]:
            for theme in themes:
                for vp in [1, 2, 3, 4]:
                    # Create 2 cards for this combination
                    for copy in range(2):
                        card = Card(
                            f"{art_type.value} {theme.value} {vp}VP-{copy+1}",
                            art_type=art_type,
                            theme=theme,
                            vp=vp
                        )
                        works.append(card)

        return works

    def _create_artist_deck(self) -> List[Card]:
        """Create the deck of 28 artist cards."""
        artists = []

        # 8 of each main type (Artesano, Pintor, Escultor)
        main_types = [
            (ArtType.CRAFTS, 8),
            (ArtType.PAINTING, 8),
            (ArtType.SCULPTURE, 8),
            (ArtType.RELIC, 4)  # 4 Anticuarios
        ]

        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        for art_type, count in main_types:
            if art_type == ArtType.RELIC:
                # Antiquarians have no theme
                for i in range(count):
                    card = Card(
                        f"Antiquarian {i+1}",
                        art_type=art_type,
                        theme=Theme.NONE
                    )
                    artists.append(card)
            else:
                # 2 of each theme for each main type
                cards_per_theme = count // 4
                type_name_map = {
                    ArtType.CRAFTS: "Craftsman",
                    ArtType.PAINTING: "Painter",
                    ArtType.SCULPTURE: "Sculptor"
                }
                for theme in themes:
                    for i in range(cards_per_theme):
                        artist_name = f"{type_name_map[art_type]} {theme.value} {i+1}"
                        card = Card(
                            artist_name,
                            art_type=art_type,
                            theme=theme
                        )
                        artists.append(card)

        return artists

    def _create_moda_tema_cards(self) -> List[Card]:
        """Create fashion theme cards (public objectives for themes)."""
        cards = []
        # Need 3 works of same theme to score 3 VP
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
        """Create fashion set cards (public objectives for adjacent sets)."""
        cards = []
        # 3 adjacent works of different themes score 3 VP
        theme_combos = [
            ([Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY], "Green-Blue-Yellow adjacents"),
            ([Theme.NATURE, Theme.MYTHOLOGY, Theme.ORIENTALISM], "Green-Blue-Red adjacents"),
            ([Theme.NATURE, Theme.SOCIETY, Theme.ORIENTALISM], "Green-Yellow-Red adjacents"),
            ([Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM], "Blue-Yellow-Red adjacents"),
        ]
        for themes, name in theme_combos:
            card = Card(
                name,
                objective_type="adjacent_set",
                required_themes=themes,
                vp=3
            )
            cards.append(card)
        return cards

    def _create_encargo_cards(self) -> List[Card]:
        """
        Create commission cards (secret objectives).

        20 total encargo cards:
        - 6 types of combinations (2 of one type + 1 of another), 3 cards each = 18 cards
        - 1 type of reliquia combination (2 reliquias), 2 cards = 2 cards
        """
        cards = []

        # All combinations of 2 and 1 of crafts, painting, sculpture (3 of each)
        combinations = [
            ("2 Crafts + 1 Painting", [ArtType.CRAFTS, ArtType.PAINTING], [2, 1], 3),
            ("2 Crafts + 1 Sculpture", [ArtType.CRAFTS, ArtType.SCULPTURE], [2, 1], 3),
            ("2 Paintings + 1 Crafts", [ArtType.PAINTING, ArtType.CRAFTS], [2, 1], 3),
            ("2 Paintings + 1 Sculpture", [ArtType.PAINTING, ArtType.SCULPTURE], [2, 1], 3),
            ("2 Sculptures + 1 Crafts", [ArtType.SCULPTURE, ArtType.CRAFTS], [2, 1], 3),
            ("2 Sculptures + 1 Painting", [ArtType.SCULPTURE, ArtType.PAINTING], [2, 1], 3),
        ]

        # Add 3 of each combination
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

        # Add 2 relic cards
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

    def _create_player_board(self, player_name: str) -> Board:
        """Create a player board matching the actual game layout with 15 spaces in 5 rooms."""
        board = Board(f"{player_name}'s House")

        # Grid layout (3 rows x 5 columns):
        # Row 1: [0] [1] [2] | [3] [4]
        # Row 2: [5] [6] [7] || [8] [9]
        # Row 3: [10] [11] | [12] [13] [14]

        # Room assignments (top to bottom, left to right):
        # Room 1: 0, 1, 2 (3 slots, top left)
        # Room 2: 3, 4, 8, 9 (4 slots, top/middle right)
        # Room 3: 5, 6, 7 (3 slots, middle left)
        # Room 4: 10, 11 (2 slots, bottom left)
        # Room 5: 12, 13, 14 (3 slots, bottom center/right)

        room_layout = [
            (0, "Room 1 (3 slots)", 1), (1, "Room 1 (3 slots)", 2), (2, "Room 1 (3 slots)", 3),
            (3, "Room 2 (4 slots)", 1), (4, "Room 2 (4 slots)", 2),
            (5, "Room 3 (3 slots)", 1), (6, "Room 3 (3 slots)", 2), (7, "Room 3 (3 slots)", 3),
            (8, "Room 2 (4 slots)", 3), (9, "Room 2 (4 slots)", 4),
            (10, "Room 4 (2 slots)", 1), (11, "Room 4 (2 slots)", 2),
            (12, "Room 5 (3 slots)", 1), (13, "Room 5 (3 slots)", 2), (14, "Room 5 (3 slots)", 3)
        ]

        # Create slots
        slots_by_id = {}
        for space_id, room_name, room_slot_num in room_layout:
            slot = Slot(f"{room_name}_{room_slot_num}", max_cards=1)
            slot.set_property("room", room_name)
            slot.set_property("space_id", space_id)
            slot.set_property("adjacent_spaces", [])  # Will populate next
            board.add_slot(slot)
            slots_by_id[space_id] = slot

        # Define door adjacencies (spaces connected by doors)
        door_connections = [
            (0, 5),   # Room 1 ↔ Room 3
            (2, 3),   # Room 1 ↔ Room 2
            (7, 12),  # Room 3 ↔ Room 5
            (11, 12), # Room 4 ↔ Room 5
            (8, 13)   # Room 2 ↔ Room 5
        ]

        # Set up adjacency lists
        for space1, space2 in door_connections:
            slots_by_id[space1].get_property("adjacent_spaces").append(space2)
            slots_by_id[space2].get_property("adjacent_spaces").append(space1)

        return board

    def play_talent_hunt_phase(self, player: ModernismePlayer) -> None:
        """Execute the talent hunt phase for a player."""
        artist_deck = self.get_deck("artists")

        # Draw new artist
        if artist_deck.is_empty():
            # Reshuffle discards
            artist_deck.cards = self.artist_discard[:-1]  # Keep top card
            artist_deck.shuffle()
            self.artist_discard = [self.artist_discard[-1]]

        new_artist = artist_deck.draw(1)[0]

        # Player chooses which artist to replace (simplified: replace first)
        old_artist = player.active_artists[0]
        player.active_artists[0] = new_artist

        # Discard old artist
        self.artist_discard.append(old_artist)

        self.log(f"  {player.name} hired {new_artist.name}, dismissed {old_artist.name}")

    def end_season(self) -> None:
        """Handle end of season: pass discards and refill hands."""
        self.log(f"\n=== End of Season {self.season} ===")

        # Pass discards counter-clockwise (to the right in player list)
        discards_to_pass = []
        for player in self.players:
            discards_to_pass.append(player.discard_pile[:])
            player.discard_pile = []

        self.log("\nPassing discards counterclockwise:")
        for i, player in enumerate(self.players):
            # Counterclockwise = next player in list (wraps around)
            next_idx = (i + 1) % len(self.players)
            cards_received = discards_to_pass[i]
            if cards_received:
                self.log(f"  {self.players[next_idx].name} receives {len(cards_received)} cards from {player.name}")
                for card in cards_received:
                    self.players[next_idx].hand.add_card(card)

        # Refill hands to 6 cards
        work_deck = self.get_deck("works")
        self.log("\nRefilling hands to 6 cards:")
        for player in self.players:
            cards_before = len(player.hand)
            while len(player.hand) < 6 and not work_deck.is_empty():
                cards = work_deck.draw(1)
                if cards:
                    player.hand.add_card(cards[0])
            cards_drawn = len(player.hand) - cards_before
            if cards_drawn > 0:
                self.log(f"  {player.name} drew {cards_drawn} cards (now has {len(player.hand)})")

        # First player marker passes clockwise (to previous player in counterclockwise order)
        self.first_player_idx = (self.first_player_idx - 1) % len(self.players)
        self.log(f"\nFirst player for next season: {self.players[self.first_player_idx].name}")

        self.season += 1

    def _check_adjacent_set(self, player: 'ModernismePlayer', required_themes: List[Theme]) -> int:
        """Check how many times a player has completed an adjacent set of themes."""
        board = self.get_board(f"{player.name}_board")

        # Get all placed works with their space IDs
        works_by_space = {}
        for slot in board.slots:
            if not slot.is_empty():
                space_id = slot.get_property("space_id")
                work = slot.get_cards()[0]
                works_by_space[space_id] = work

        # Find all sets of 3 adjacent works with the required themes
        count = 0
        checked_sets = set()

        for space_id, work in works_by_space.items():
            slot = next(s for s in board.slots if s.get_property("space_id") == space_id)
            adjacent_spaces = slot.get_property("adjacent_spaces")

            # Check all combinations of this work and 2 adjacent works
            for adj1 in adjacent_spaces:
                if adj1 not in works_by_space:
                    continue
                for adj2 in adjacent_spaces:
                    if adj2 not in works_by_space or adj2 == adj1:
                        continue

                    # Check if these 3 works form a valid set
                    works = [work, works_by_space[adj1], works_by_space[adj2]]
                    # Use assigned themes for Reliquias
                    themes = set()
                    for w in works:
                        if w.get_property("art_type") == ArtType.RELIC and w in player.reliquia_themes:
                            themes.add(player.reliquia_themes[w])
                        else:
                            themes.add(w.get_property("theme"))

                    # Must have all 3 required themes
                    if themes == set(required_themes):
                        # Create a unique identifier for this set (sorted space IDs)
                        set_id = tuple(sorted([space_id, adj1, adj2]))
                        if set_id not in checked_sets:
                            checked_sets.add(set_id)
                            count += 1

        return count

    def _score_objectives(self):
        """Score all moda and encargo objectives."""
        self.log("\n" + "=" * 60)
        self.log("OBJECTIVE SCORING - Moda & Encargo")
        self.log("=" * 60)

        for player in self.players:
            board = self.get_board(f"{player.name}_board")
            total_objective_vp = 0

            self.log(f"\n{player.name}:")

            # Count works by type and theme
            works = []
            for slot in board.slots:
                if not slot.is_empty():
                    works.append(slot.get_cards()[0])

            type_counts = {}
            theme_counts = {}
            for work in works:
                art_type = work.get_property("art_type")
                # Use assigned theme for Reliquias
                if art_type == ArtType.RELIC and work in player.reliquia_themes:
                    theme = player.reliquia_themes[work]
                else:
                    theme = work.get_property("theme")
                type_counts[art_type] = type_counts.get(art_type, 0) + 1
                if theme != Theme.NONE:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1

            # Score moda_tema (public theme objective)
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

            # Score moda_conjunto (public adjacent set objective)
            if self.moda_conjunto:
                required_themes = self.moda_conjunto.get_property("required_themes")
                vp_per = self.moda_conjunto.get_property("vp")

                times_completed = self._check_adjacent_set(player, required_themes)
                if times_completed > 0:
                    vp_earned = times_completed * vp_per
                    total_objective_vp += vp_earned
                    self.log(f"  Set Fashion ({self.moda_conjunto.name}): {times_completed}x = +{vp_earned} VP")

            # Score encargo (secret commission)
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
                    # Check if all requirements are met
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


def play_modernisme_game(log_file: Optional[TextIO] = None, num_players: int = 4):
    """
    Play a complete game of Modernisme.

    Args:
        log_file: Optional file to write game log to
        num_players: Number of players (default 4)

    Returns:
        ModernismeGame: The completed game instance
    """
    game = ModernismeGame(log_file=log_file)

    game.log("=" * 60)
    game.log("MODERNISME - Barcelona's Modernist Movement")
    game.log("=" * 60)

    game.setup_game(num_players=num_players)

    # Play 4 seasons
    for season in range(1, 5):
        game.log(f"\n{'='*60}")
        game.log(f"SEASON {season}")
        game.log(f"First player: {game.players[game.first_player_idx].name}")
        game.log('='*60)

        # Each player takes a turn in counterclockwise order from first player
        for offset in range(len(game.players)):
            player_idx = (game.first_player_idx + offset) % len(game.players)
            player = game.players[player_idx]

            game.log(f"\n--- {player.name}'s Turn ---")

            # Show hand at start of turn
            game.log("Hand at start of turn:")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            game.log(f"  {', '.join(hand_cards)}")

            # Phase 1: Talent Hunt
            game.log("\nPhase 1: Talent Hunt")
            game.play_talent_hunt_phase(player)

            # Phase 2: Placement
            game.log("Phase 2: Commissioning Works")
            game.log(f"  Active artists: {[a.name for a in player.active_artists]}")
            player.strategy(game)

            # Show hand at end of turn
            game.log(f"\nHand at end of turn ({len(player.hand)} cards):")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            if hand_cards:
                game.log(f"  {', '.join(hand_cards)}")
            else:
                game.log(f"  (empty)")
            game.log(f"Score: {player.score} VP")

        # End of season
        if season < 4:
            game.end_season()

    # Contextualize Relics before scoring
    game.log("\n" + "=" * 60)
    game.log("CONTEXTUALIZING RELICS")
    game.log("=" * 60)
    for player in game.players:
        player.assign_reliquia_themes(game)

    # Score objectives (moda and encargo cards)
    game._score_objectives()

    # Final scoring - Room completion bonuses
    game.log("\n" + "=" * 60)
    game.log("ROOM COMPLETION BONUSES")
    game.log("=" * 60)

    for player in game.players:
        board = game.get_board(f"{player.name}_board")
        game.log(f"\n{player.name}:")

        # Check completed rooms
        rooms = {}
        for slot in board.slots:
            room_name = slot.get_property("room")
            if room_name not in rooms:
                rooms[room_name] = []
            if not slot.is_empty():
                rooms[room_name].append(slot.get_cards()[0])

        # Room bonuses
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
                works = rooms[room_name]

                # Check if all same type
                types = [w.get_property("art_type") for w in works]
                same_type = len(set(types)) == 1

                # Check if all same theme (use assigned themes for Reliquias)
                themes = []
                for w in works:
                    if w.get_property("art_type") == ArtType.RELIC and w in player.reliquia_themes:
                        themes.append(player.reliquia_themes[w])
                    elif w.get_property("theme") != Theme.NONE:
                        themes.append(w.get_property("theme"))
                same_theme = len(themes) == required_works and len(set(themes)) == 1

                # Bonuses are NOT cumulative - only count one per room
                # Prefer same type bonus if both conditions are met
                if same_type:
                    bonus = required_works
                    player.add_score(bonus)
                    room_status += f" → same type: +{bonus} VP"
                elif same_theme:
                    bonus = required_works
                    player.add_score(bonus)
                    room_status += f" → same theme: +{bonus} VP"

            game.log(room_status)

    # Determine winner
    game.log("\n" + "=" * 60)
    game.log("FINAL SCORES")
    game.log("=" * 60)

    for player in game.players:
        board = game.get_board(f"{player.name}_board")
        total_works = sum(1 for slot in board.slots if not slot.is_empty())
        game.log(f"{player.name}: {player.score} VP ({total_works} works placed)")

    winner = max(game.players, key=lambda p: (p.score, -sum(1 for slot in game.get_board(f"{p.name}_board").slots if not slot.is_empty())))

    game.log("\n" + "=" * 60)
    game.log(f"🏆 WINNER: {winner.name} with {winner.score} VP!")
    game.log("=" * 60)

    return game


if __name__ == "__main__":
    play_modernisme_game()
