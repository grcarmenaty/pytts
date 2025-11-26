"""Basic tests for PyTTS framework."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Card, Hand, Deck, DiscardPile, Slot, Board, Player, Game


def test_card():
    """Test card creation and properties."""
    card = Card("Ace of Spades", value=14, suit="Spades")
    assert card.name == "Ace of Spades"
    assert card.get_property("value") == 14
    assert card.get_property("suit") == "Spades"
    card.set_property("color", "black")
    assert card.get_property("color") == "black"
    print("✓ Card tests passed")


def test_hand():
    """Test hand operations."""
    hand = Hand(max_size=5)
    card1 = Card("Card 1")
    card2 = Card("Card 2")

    assert hand.add_card(card1)
    assert hand.add_card(card2)
    assert hand.size() == 2
    assert not hand.is_empty()

    assert hand.remove_card(card1)
    assert hand.size() == 1
    print("✓ Hand tests passed")


def test_deck():
    """Test deck operations."""
    cards = [Card(f"Card {i}") for i in range(10)]
    deck = Deck(cards)

    assert deck.size() == 10
    drawn = deck.draw(3)
    assert len(drawn) == 3
    assert deck.size() == 7

    deck.shuffle()
    assert deck.size() == 7

    hand = Hand()
    deck.draw_to_hand(hand, 5)
    assert hand.size() == 5
    assert deck.size() == 2
    print("✓ Deck tests passed")


def test_discard_pile():
    """Test discard pile."""
    discard = DiscardPile()
    card = Card("Discarded Card")

    discard.add_card(card)
    assert discard.is_visible_to_all()
    all_cards = discard.get_all_cards()
    assert len(all_cards) == 1
    assert all_cards[0] == card
    print("✓ Discard pile tests passed")


def test_slot_and_board():
    """Test slot and board operations."""
    def can_place_even(card, player):
        return card.get_property("value", 0) % 2 == 0

    slot = Slot("test_slot", can_place_rule=can_place_even)
    card_even = Card("2", value=2)
    card_odd = Card("3", value=3)

    player = Player("Test Player")

    assert slot.can_place_card(card_even, player)
    assert not slot.can_place_card(card_odd, player)

    slot.place_card(card_even, player)
    assert not slot.is_empty()
    assert slot.get_cards()[0] == card_even

    board = Board("test_board")
    board.add_slot(slot)
    assert board.get_slot("test_slot") == slot
    print("✓ Slot and Board tests passed")


def test_player():
    """Test player operations."""
    player = Player("Alice", hand_max_size=5)
    assert player.name == "Alice"
    assert player.is_active
    assert player.score == 0

    deck = Deck([Card(f"Card {i}") for i in range(10)])
    player.draw_cards(deck, 3)
    assert player.hand.size() == 3

    player.add_score(10)
    assert player.score == 10
    print("✓ Player tests passed")


def test_game():
    """Test game setup and execution."""
    game = Game("Test Game")

    player1 = Player("Alice")
    player2 = Player("Bob")
    game.add_player(player1)
    game.add_player(player2)

    deck = Deck([Card(f"Card {i}") for i in range(20)])
    game.add_deck("main", deck)

    board = Board("main")
    game.add_board("main", board)

    assert len(game.players) == 2
    assert game.get_deck("main") == deck
    assert game.get_board("main") == board

    win_condition_called = [False]

    def win_condition(g):
        win_condition_called[0] = True
        return False

    game.set_win_condition(win_condition)
    game.run(max_rounds=2)

    assert game.round_number == 2
    assert win_condition_called[0]
    print("✓ Game tests passed")


def run_all_tests():
    """Run all tests."""
    print("\nRunning PyTTS Tests")
    print("=" * 50)

    test_card()
    test_hand()
    test_deck()
    test_discard_pile()
    test_slot_and_board()
    test_player()
    test_game()

    print("=" * 50)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
