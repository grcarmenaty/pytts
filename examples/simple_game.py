"""
Simple card game example using PyTTS framework.

This example demonstrates a basic "High Card" game where:
- Each player draws a card each turn
- Players play their card to a slot
- Highest card value wins the round
- First to 3 points wins
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Game, Player, Card, Deck, Board, Slot


class HighCardPlayer(Player):
    """Player that plays the highest card from their hand."""

    def strategy(self, game: Game) -> None:
        """Play the highest value card."""
        deck = game.get_deck("main")
        board = game.get_board("main")

        if deck and not deck.is_empty():
            self.draw_cards(deck, 1)

        if self.hand.is_empty():
            return

        highest_card = max(self.hand.cards, key=lambda c: c.get_property("value", 0))

        slot = board.get_slot(f"{self.name}_slot")
        if slot:
            self.play_card(highest_card, slot)


def create_standard_deck():
    """Create a standard deck of numbered cards."""
    cards = []
    for value in range(1, 14):
        for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            card = Card(f"{value} of {suit}", value=value, suit=suit)
            cards.append(card)
    return cards


def setup_game():
    """Set up the game."""
    game = Game("High Card Game")

    player1 = HighCardPlayer("Alice")
    player2 = HighCardPlayer("Bob")
    game.add_player(player1)
    game.add_player(player2)

    deck = Deck(create_standard_deck())
    deck.shuffle()
    game.add_deck("main", deck)

    board = Board("main")
    board.add_slot(Slot("Alice_slot"))
    board.add_slot(Slot("Bob_slot"))
    game.add_board("main", board)

    def check_winner(g: Game) -> bool:
        """Check if any player has won."""
        for player in g.players:
            if player.score >= 3:
                g.end_game(player)
                return True
        return False

    game.set_win_condition(check_winner)

    return game


def evaluate_round(game: Game):
    """Evaluate who won the round."""
    board = game.get_board("main")
    if not board:
        return

    alice_slot = board.get_slot("Alice_slot")
    bob_slot = board.get_slot("Bob_slot")

    if not alice_slot or not bob_slot:
        return

    alice_cards = alice_slot.get_cards()
    bob_cards = bob_slot.get_cards()

    if not alice_cards or not bob_cards:
        return

    alice_value = alice_cards[0].get_property("value", 0)
    bob_value = bob_cards[0].get_property("value", 0)

    alice = game.get_player_by_name("Alice")
    bob = game.get_player_by_name("Bob")

    print(f"  Alice played: {alice_cards[0]} (value: {alice_value})")
    print(f"  Bob played: {bob_cards[0]} (value: {bob_value})")

    if alice_value > bob_value:
        alice.add_score(1)
        print(f"  -> Alice wins this round!")
    elif bob_value > alice_value:
        bob.add_score(1)
        print(f"  -> Bob wins this round!")
    else:
        print(f"  -> Tie!")

    alice_slot.clear()
    bob_slot.clear()


def main():
    """Run the game."""
    print("=" * 50)
    print("High Card Game - First to 3 points wins!")
    print("=" * 50)

    game = setup_game()

    round_num = 0
    while not game.is_finished and round_num < 10:
        round_num += 1
        print(f"\nRound {round_num}:")

        game.play_round()

        evaluate_round(game)

        alice = game.get_player_by_name("Alice")
        bob = game.get_player_by_name("Bob")
        print(f"  Score - Alice: {alice.score}, Bob: {bob.score}")

        game.check_win_condition()

    print("\n" + "=" * 50)
    if game.winner:
        print(f"Game Over! {game.winner.name} wins!")
    else:
        print("Game Over! No winner.")
    print("=" * 50)


if __name__ == "__main__":
    main()
