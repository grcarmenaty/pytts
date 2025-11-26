# PyTTS - Python Tabletop Simulator

An object-oriented framework for building tabletop game probability simulators in Python.

## Features

- **Flexible OOP Design**: Clean class hierarchy for building complex card games
- **Strategy Pattern**: Implement custom player strategies for game simulation
- **Rule-Based Slots**: Define placement rules for board positions
- **Multiple Decks**: Support for draw decks, discard piles, and custom deck types
- **Win Conditions**: Customizable game-ending conditions
- **Round Management**: Automatic turn and round execution

## Architecture

### Core Classes

- **Card**: Basic building block with properties and methods
- **Hand**: Manages a player's collection of cards
- **Deck**: Card collection with draw and shuffle functionality
- **DiscardPile**: Special visible deck for discarded cards
- **Slot**: Board position with placement/removal rules
- **Board**: Collection of slots
- **Player**: Has hand and implements strategy pattern
- **Round**: Manages turn execution
- **Game**: Orchestrates all game elements

### Class Hierarchy

```
Game
├── Players (with strategy methods)
│   └── Hand (collection of Cards)
├── Decks (draw, shuffle)
│   └── Cards
├── Boards
│   └── Slots (with rules)
│       └── Cards
└── Rounds
    └── Turns (execute player strategies)
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd pytts

# No dependencies required - pure Python!
```

## Quick Start

```python
from pytts import Game, Player, Card, Deck, Board, Slot

# Create a custom player with strategy
class MyPlayer(Player):
    def strategy(self, game):
        # Draw a card
        deck = game.get_deck("main")
        self.draw_cards(deck, 1)

        # Play a card
        if not self.hand.is_empty():
            card = self.hand.cards[0]
            slot = game.get_board("main").get_slot("slot1")
            self.play_card(card, slot)

# Set up the game
game = Game("My Game")
game.add_player(MyPlayer("Alice"))
game.add_player(MyPlayer("Bob"))

# Create and add a deck
cards = [Card(f"Card {i}", value=i) for i in range(10)]
deck = Deck(cards)
deck.shuffle()
game.add_deck("main", deck)

# Create a board with slots
board = Board("main")
board.add_slot(Slot("slot1"))
board.add_slot(Slot("slot2"))
game.add_board("main", board)

# Run the game
game.run(max_rounds=5)
```

## Examples

See `examples/simple_game.py` for a complete "High Card" game implementation.

```bash
python examples/simple_game.py
```

## Testing

Run the basic test suite:

```bash
python tests/test_basic.py
```

## Advanced Features

### Custom Slot Rules

```python
def can_place_rule(card, player):
    # Only allow cards with even values
    return card.get_property("value", 0) % 2 == 0

def can_take_rule(card, player):
    # Only owner can take cards
    return card.owner == player

slot = Slot("restricted_slot",
            can_place_rule=can_place_rule,
            can_take_rule=can_take_rule,
            max_cards=3)
```

### Win Conditions

```python
def check_score_win(game):
    for player in game.players:
        if player.score >= 10:
            game.end_game(player)
            return True
    return False

game.set_win_condition(check_score_win)
```

### Discard Piles

```python
from pytts import DiscardPile

discard = DiscardPile()
game.add_discard_pile("discard", discard)

# All players can see all cards
all_discarded = discard.get_all_cards()
```

## Use Cases

- **Probability Analysis**: Simulate thousands of games to analyze strategies
- **Game Design**: Prototype and test game mechanics
- **AI Development**: Train agents with different strategies
- **Educational**: Learn OOP design patterns through game development

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! This is a framework designed for extensibility.
