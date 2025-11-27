"""
Modernisme - A complete implementation of the tabletop game.

This example implements the full Modernisme game based on the rules v0.4.1,
set in Barcelona's Modernist movement of the 19th century.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Game, Player, Card, Deck, Board, Slot, Hand, DiscardPile
import random
from typing import List, Optional, Tuple
from enum import Enum


class ArtType(Enum):
    """Types of art/artists in the game."""
    CRAFTS = "Artesanía"  # Artesano
    PAINTING = "Pintura"  # Pintor
    SCULPTURE = "Escultura"  # Escultor
    RELIC = "Reliquia"  # Anticuario


class Theme(Enum):
    """Themes for works and artists."""
    NATURE = "Naturaleza"  # Green
    MYTHOLOGY = "Mitología"  # Blue
    SOCIETY = "Costumbrismo"  # Yellow
    ORIENTALISM = "Orientalismo"  # Red
    NONE = "Sin tema"  # For relics


class ModernismePlayer(Player):
    """Player in Modernisme game."""

    def __init__(self, name: str):
        super().__init__(name, hand_max_size=None)  # No hand limit
        self.active_artists: List[Card] = []  # 2 active artist cards
        self.discard_pile: List[Card] = []  # Player's work card discards
        self.commission_card: Optional[Card] = None  # Secret objective
        self.completed_rooms: List[int] = []  # Track which rooms are complete

    def strategy(self, game: 'ModernismeGame') -> None:
        """
        AI strategy for Modernisme.

        Simple strategy:
        1. Commission works that match active artists
        2. Try to complete rooms with same theme or type
        3. Focus on higher VP works
        """
        # Simple AI: commission 1-2 works per turn
        works_to_commission = min(2, len(self.hand.cards))

        for _ in range(works_to_commission):
            if not self.hand.cards:
                break

            # Find a work we can commission
            for work in self.hand.cards[:]:
                work_type = work.get_property("art_type")

                # Check if we have an artist for this work type
                can_commission = any(
                    artist.get_property("art_type") == work_type
                    for artist in self.active_artists
                )

                if can_commission and self._try_commission_work(work, game):
                    break

    def _try_commission_work(self, work: Card, game: 'ModernismeGame') -> bool:
        """Try to commission a specific work, minimizing overspending."""
        work_vp = work.get_property("vp", 0)

        # Get available cards for discarding (excluding the work itself)
        available_cards = [c for c in self.hand.cards if c != work]

        if not available_cards:
            return False

        # Find the best combination of cards to discard that minimizes overspending
        # Sort cards by VP value
        card_vps = [(card, card.get_property("vp", 0)) for card in available_cards]
        card_vps.sort(key=lambda x: x[1])

        best_discard = None
        best_overspend = float('inf')

        # Try to find combinations that meet the requirement
        # Use a greedy approach: try starting with larger cards first
        for i in range(len(card_vps)):
            discard_cards = []
            total_vp = 0

            # Start with current card and add smaller cards as needed
            for j in range(i, len(card_vps)):
                card, vp = card_vps[j]
                discard_cards.append(card)
                total_vp += vp

                if total_vp >= work_vp:
                    overspend = total_vp - work_vp
                    if overspend < best_overspend:
                        best_overspend = overspend
                        best_discard = discard_cards[:]
                    break

        # Also try greedy approach from smallest to largest
        discard_cards = []
        total_vp = 0
        for card, vp in card_vps:
            if total_vp < work_vp:
                discard_cards.append(card)
                total_vp += vp

        if total_vp >= work_vp:
            overspend = total_vp - work_vp
            if overspend < best_overspend:
                best_overspend = overspend
                best_discard = discard_cards[:]

        if best_discard is None:
            return False

        # Log discarded cards
        total_vp = sum(card.get_property("vp", 0) for card in best_discard)
        discard_names = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in best_discard]
        print(f"    Discarded: {', '.join(discard_names)} (total {total_vp} VP for {work_vp} VP work, overspend: {best_overspend} VP)")

        # Discard cards
        for card in best_discard:
            self.hand.remove_card(card)
            self.discard_pile.append(card)

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
                    print(f"    → Placed '{work.name}' in {room_name} (space {space_id})")
                    print(f"      Base VP: {base_vp}", end="")
                    if bonus_vp > 0:
                        print(f" + {bonus_vp} (theme bonus from {matching_artist.name})", end="")
                    print(f" = {vp_gain} VP total")
                    print(f"      {self.name}'s total score: {self.score} VP")

                    return True

        return False


class ModernismeGame(Game):
    """The Modernisme game implementation."""

    def __init__(self):
        super().__init__("Modernisme")
        self.artist_discard: List[Card] = []  # Artist discard pile
        self.season = 1  # Current season (1-4)
        self.first_player_idx = 0  # Index of first player
        self.moda_tema: Optional[Card] = None  # Public theme fashion card
        self.moda_conjunto: Optional[Card] = None  # Public set fashion card

    def setup_game(self, num_players: int = 2):
        """Set up a game of Modernisme."""
        # Create players
        player_names = ["Alice", "Bob", "Carol", "David"]
        for i in range(num_players):
            player = ModernismePlayer(player_names[i])
            self.add_player(player)

        # Select moda cards (public objectives)
        moda_tema_cards = self._create_moda_tema_cards()
        moda_conjunto_cards = self._create_moda_conjunto_cards()
        self.moda_tema = random.choice(moda_tema_cards)
        self.moda_conjunto = random.choice(moda_conjunto_cards)

        print(f"\nPublic Objectives:")
        print(f"  Theme Fashion: {self.moda_tema.name}")
        print(f"  Set Fashion: {self.moda_conjunto.name}")

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
        print(f"\nPlayers selecting secret commissions:")
        for player in self.players:
            # Draw 3, pick 1
            options = encargo_cards[:3]
            encargo_cards = encargo_cards[3:]  # Remove used cards
            # AI randomly picks one
            player.commission_card = random.choice(options)
            print(f"  {player.name} selected a secret commission")

        print(f"\nGame setup complete with {num_players} players!")
        print(f"Each player has 6 work cards, 2 active artists, and 1 secret commission.")

    def _create_work_deck(self) -> List[Card]:
        """Create the deck of 112 work cards matching xlsx specifications."""
        works = []

        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        # Relics: 16 cards, all 5 VP
        for i in range(16):
            card = Card(
                f"Reliquia {i+1}",
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
                        f"Anticuario {i+1}",
                        art_type=art_type,
                        theme=Theme.NONE
                    )
                    artists.append(card)
            else:
                # 2 of each theme for each main type
                cards_per_theme = count // 4
                for theme in themes:
                    for i in range(cards_per_theme):
                        artist_name = f"{art_type.value.replace('ía', 'o').replace('ura', 'or')} {theme.value} {i+1}"
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
        """Create commission cards (secret objectives)."""
        cards = []
        # Type-based objectives (3 cards)
        for art_type in [ArtType.CRAFTS, ArtType.PAINTING, ArtType.SCULPTURE]:
            card = Card(
                f"3 {art_type.value} works",
                objective_type="type_count",
                required_type=art_type,
                required_count=3,
                vp=3
            )
            cards.append(card)

        # Theme-based objectives (4 cards)
        for theme in [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]:
            card = Card(
                f"4 {theme.value} works",
                objective_type="theme_count",
                required_theme=theme,
                required_count=4,
                vp=3
            )
            cards.append(card)

        # Mixed objectives (3 cards to total 10 encargo cards)
        mixed_objectives = [
            ("2 Reliquias + 2 Esculturas", [ArtType.RELIC, ArtType.SCULPTURE], [2, 2], 3),
            ("2 Reliquias + 2 Pinturas", [ArtType.RELIC, ArtType.PAINTING], [2, 2], 3),
            ("2 Artesanías + 2 Pinturas", [ArtType.CRAFTS, ArtType.PAINTING], [2, 2], 3),
        ]

        for name, req_types, req_counts, vp in mixed_objectives:
            card = Card(
                name,
                objective_type="mixed",
                required_types=req_types,
                required_counts=req_counts,
                vp=vp
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

        # Room assignments:
        # Salón: 0, 1, 2 (3 spaces, top left)
        # Comedor: 5, 6, 7 (3 spaces, middle left)
        # Biblioteca: 3, 4, 8, 9 (4 spaces, top/middle right)
        # Dormitorio: 10, 11 (2 spaces, bottom left)
        # Vestíbulo: 12, 13, 14 (3 spaces, bottom right/center)

        room_layout = [
            (0, "Salón", 1), (1, "Salón", 2), (2, "Salón", 3),
            (3, "Biblioteca", 1), (4, "Biblioteca", 2),
            (5, "Comedor", 1), (6, "Comedor", 2), (7, "Comedor", 3),
            (8, "Biblioteca", 3), (9, "Biblioteca", 4),
            (10, "Dormitorio", 1), (11, "Dormitorio", 2),
            (12, "Vestíbulo", 1), (13, "Vestíbulo", 2), (14, "Vestíbulo", 3)
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
            (0, 5),   # Salón ↔ Comedor
            (2, 3),   # Salón ↔ Biblioteca
            (7, 12),  # Comedor ↔ Vestíbulo
            (11, 12), # Dormitorio ↔ Vestíbulo
            (8, 13)   # Biblioteca ↔ Vestíbulo
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

        print(f"  {player.name} hired {new_artist.name}, dismissed {old_artist.name}")

    def end_season(self) -> None:
        """Handle end of season: pass discards and refill hands."""
        print(f"\n=== End of Season {self.season} ===")

        # Pass discards counter-clockwise (to the right in player list)
        discards_to_pass = []
        for player in self.players:
            discards_to_pass.append(player.discard_pile[:])
            player.discard_pile = []

        print("\nPassing discards counterclockwise:")
        for i, player in enumerate(self.players):
            # Counterclockwise = next player in list (wraps around)
            next_idx = (i + 1) % len(self.players)
            cards_received = discards_to_pass[i]
            if cards_received:
                print(f"  {self.players[next_idx].name} receives {len(cards_received)} cards from {player.name}")
                for card in cards_received:
                    self.players[next_idx].hand.add_card(card)

        # Refill hands to 6 cards
        work_deck = self.get_deck("works")
        print("\nRefilling hands to 6 cards:")
        for player in self.players:
            cards_before = len(player.hand)
            while len(player.hand) < 6 and not work_deck.is_empty():
                cards = work_deck.draw(1)
                if cards:
                    player.hand.add_card(cards[0])
            cards_drawn = len(player.hand) - cards_before
            if cards_drawn > 0:
                print(f"  {player.name} drew {cards_drawn} cards (now has {len(player.hand)})")

        # First player marker passes clockwise (to previous player in counterclockwise order)
        self.first_player_idx = (self.first_player_idx - 1) % len(self.players)
        print(f"\nFirst player for next season: {self.players[self.first_player_idx].name}")

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
                    themes = set(w.get_property("theme") for w in works)

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
        print("\n" + "=" * 60)
        print("OBJECTIVE SCORING - Moda & Encargo")
        print("=" * 60)

        for player in self.players:
            board = self.get_board(f"{player.name}_board")
            total_objective_vp = 0

            print(f"\n{player.name}:")

            # Count works by type and theme
            works = []
            for slot in board.slots:
                if not slot.is_empty():
                    works.append(slot.get_cards()[0])

            type_counts = {}
            theme_counts = {}
            for work in works:
                art_type = work.get_property("art_type")
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
                    print(f"  Theme Fashion ({self.moda_tema.name}): {times_completed}x = +{vp_earned} VP")

            # Score moda_conjunto (public adjacent set objective)
            if self.moda_conjunto:
                required_themes = self.moda_conjunto.get_property("required_themes")
                vp_per = self.moda_conjunto.get_property("vp")

                times_completed = self._check_adjacent_set(player, required_themes)
                if times_completed > 0:
                    vp_earned = times_completed * vp_per
                    total_objective_vp += vp_earned
                    print(f"  Set Fashion ({self.moda_conjunto.name}): {times_completed}x = +{vp_earned} VP")

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
                    print(f"  Secret Commission ({encargo.name}): {times_completed}x = +{vp_earned} VP")
                else:
                    print(f"  Secret Commission ({encargo.name}): not completed")

            if total_objective_vp > 0:
                player.add_score(total_objective_vp)


def play_modernisme_game():
    """Play a complete game of Modernisme."""
    print("=" * 60)
    print("MODERNISME - Barcelona's Modernist Movement")
    print("=" * 60)

    game = ModernismeGame()
    game.setup_game(num_players=4)

    # Play 4 seasons
    for season in range(1, 5):
        print(f"\n{'='*60}")
        print(f"SEASON {season}")
        print(f"First player: {game.players[game.first_player_idx].name}")
        print('='*60)

        # Each player takes a turn in counterclockwise order from first player
        for offset in range(len(game.players)):
            player_idx = (game.first_player_idx + offset) % len(game.players)
            player = game.players[player_idx]

            print(f"\n--- {player.name}'s Turn ---")

            # Show hand at start of turn
            print("Hand at start of turn:")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            print(f"  {', '.join(hand_cards)}")

            # Phase 1: Talent Hunt
            print("\nPhase 1: Talent Hunt")
            game.play_talent_hunt_phase(player)

            # Phase 2: Placement
            print("Phase 2: Commissioning Works")
            print(f"  Active artists: {[a.name for a in player.active_artists]}")
            player.strategy(game)

            # Show hand at end of turn
            print(f"\nHand at end of turn ({len(player.hand)} cards):")
            hand_cards = [f"{card.name} ({card.get_property('vp', 0)} VP)" for card in player.hand.cards]
            if hand_cards:
                print(f"  {', '.join(hand_cards)}")
            else:
                print(f"  (empty)")
            print(f"Score: {player.score} VP")

        # End of season
        if season < 4:
            game.end_season()

    # Score objectives (moda and encargo cards)
    game._score_objectives()

    # Final scoring - Room completion bonuses
    print("\n" + "=" * 60)
    print("ROOM COMPLETION BONUSES")
    print("=" * 60)

    for player in game.players:
        board = game.get_board(f"{player.name}_board")
        print(f"\n{player.name}:")

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
            "Dormitorio": 2,
            "Salón": 3,
            "Comedor": 3,
            "Biblioteca": 4,
            "Vestíbulo": 3
        }

        for room_name, required_works in room_configs.items():
            works_in_room = rooms.get(room_name, [])
            print(f"  {room_name}: {len(works_in_room)}/{required_works} works", end="")

            if len(works_in_room) == required_works:
                print(" (COMPLETE)", end="")
                works = rooms[room_name]

                # Check if all same type
                types = [w.get_property("art_type") for w in works]
                same_type = len(set(types)) == 1

                # Check if all same theme
                themes = [w.get_property("theme") for w in works if w.get_property("theme") != Theme.NONE]
                same_theme = len(themes) == required_works and len(set(themes)) == 1

                # Bonuses are NOT cumulative - only count one per room
                # Prefer same type bonus if both conditions are met
                if same_type:
                    bonus = required_works
                    player.add_score(bonus)
                    print(f" → same type: +{bonus} VP")
                elif same_theme:
                    bonus = required_works
                    player.add_score(bonus)
                    print(f" → same theme: +{bonus} VP")
                else:
                    print()
            else:
                print()

    # Determine winner
    print("\n" + "=" * 60)
    print("FINAL SCORES")
    print("=" * 60)

    for player in game.players:
        board = game.get_board(f"{player.name}_board")
        total_works = sum(1 for slot in board.slots if not slot.is_empty())
        print(f"{player.name}: {player.score} VP ({total_works} works placed)")

    winner = max(game.players, key=lambda p: (p.score, -sum(1 for slot in game.get_board(f"{p.name}_board").slots if not slot.is_empty())))

    print("\n" + "=" * 60)
    print(f"🏆 WINNER: {winner.name} with {winner.score} VP!")
    print("=" * 60)


if __name__ == "__main__":
    play_modernisme_game()
