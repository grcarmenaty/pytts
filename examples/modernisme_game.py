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
    SOCIETY = "Sociedad"  # Yellow
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
        """Try to commission a specific work."""
        work_vp = work.get_property("vp", 0)

        # Find cards to discard (must sum >= work VP)
        discard_cards = []
        total_vp = 0

        for card in self.hand.cards:
            if card == work:
                continue
            discard_cards.append(card)
            total_vp += card.get_property("vp", 0)
            if total_vp >= work_vp:
                break

        if total_vp < work_vp:
            return False

        # Discard cards
        for card in discard_cards:
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
                    print(f"    → Placed '{work.name}' in {room_name}")
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

    def setup_game(self, num_players: int = 2):
        """Set up a game of Modernisme."""
        # Create players
        player_names = ["Alice", "Bob", "Carol", "David"]
        for i in range(num_players):
            player = ModernismePlayer(player_names[i])
            self.add_player(player)

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

        print(f"Game setup complete with {num_players} players!")
        print(f"Each player has 6 work cards and 2 active artists.")

    def _create_work_deck(self) -> List[Card]:
        """Create the deck of 112 work cards."""
        works = []

        # Define work types and themes
        art_types = [
            (ArtType.CRAFTS, 32),
            (ArtType.PAINTING, 32),
            (ArtType.SCULPTURE, 32),
            (ArtType.RELIC, 16)
        ]

        themes = [Theme.NATURE, Theme.MYTHOLOGY, Theme.SOCIETY, Theme.ORIENTALISM]

        # Create cards for each type
        for art_type, count in art_types:
            cards_per_theme = count // 4 if art_type != ArtType.RELIC else 0

            if art_type == ArtType.RELIC:
                # Relics have no theme
                for i in range(count):
                    vp = random.choice([2, 3, 4, 5])  # Simplified VP distribution
                    card = Card(
                        f"Reliquia {i+1}",
                        art_type=art_type,
                        theme=Theme.NONE,
                        vp=vp
                    )
                    works.append(card)
            else:
                # Regular works have themes
                for theme in themes:
                    for i in range(cards_per_theme):
                        vp = random.choice([1, 2, 2, 3, 3, 4])  # Simplified distribution
                        card = Card(
                            f"{art_type.value} {theme.value} {i+1}",
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

    def _create_player_board(self, player_name: str) -> Board:
        """Create a player board with 4 rooms."""
        board = Board(f"{player_name}'s House")

        # Create 4 rooms with different sizes
        room_configs = [
            ("Dormitorio", 2),
            ("Salón", 3),
            ("Comedor", 3),
            ("Biblioteca", 4)
        ]

        for room_name, num_slots in room_configs:
            for i in range(num_slots):
                slot = Slot(f"{room_name}_{i+1}", max_cards=1)
                slot.set_property("room", room_name)
                board.add_slot(slot)

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

        # Pass discards counter-clockwise
        discards_to_pass = []
        for player in self.players:
            discards_to_pass.append(player.discard_pile[:])
            player.discard_pile = []

        for i, player in enumerate(self.players):
            next_idx = (i - 1) % len(self.players)
            for card in discards_to_pass[next_idx]:
                player.hand.add_card(card)

        # Refill hands to 6 cards
        work_deck = self.get_deck("works")
        for player in self.players:
            while len(player.hand) < 6 and not work_deck.is_empty():
                cards = work_deck.draw(1)
                if cards:
                    player.hand.add_card(cards[0])

        self.season += 1


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
        print('='*60)

        # Each player takes a turn
        for player in game.players:
            print(f"\n--- {player.name}'s Turn ---")

            # Phase 1: Talent Hunt
            print("Phase 1: Talent Hunt")
            game.play_talent_hunt_phase(player)

            # Phase 2: Placement
            print("Phase 2: Commissioning Works")
            print(f"  Active artists: {[a.name for a in player.active_artists]}")
            print(f"  Hand size: {len(player.hand)} cards")
            player.strategy(game)
            print(f"  Score: {player.score} VP")

        # End of season
        if season < 4:
            game.end_season()

    # Final scoring
    print("\n" + "=" * 60)
    print("FINAL SCORING - Room Completion Bonuses")
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

        # Room bonuses (simplified)
        room_configs = {"Dormitorio": 2, "Salón": 3, "Comedor": 3, "Biblioteca": 4}

        for room_name, required_works in room_configs.items():
            works_in_room = rooms.get(room_name, [])
            print(f"  {room_name}: {len(works_in_room)}/{required_works} works", end="")

            if len(works_in_room) == required_works:
                print(" (COMPLETE)", end="")
                works = rooms[room_name]
                bonuses_earned = []

                # Check if all same type
                types = [w.get_property("art_type") for w in works]
                if len(set(types)) == 1:
                    bonus = required_works
                    player.add_score(bonus)
                    bonuses_earned.append(f"same type: +{bonus} VP")

                # Check if all same theme
                themes = [w.get_property("theme") for w in works if w.get_property("theme") != Theme.NONE]
                if len(themes) == required_works and len(set(themes)) == 1:
                    bonus = required_works
                    player.add_score(bonus)
                    bonuses_earned.append(f"same theme: +{bonus} VP")

                if bonuses_earned:
                    print(f" → {', '.join(bonuses_earned)}")
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
