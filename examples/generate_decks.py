"""
Generate all Modernisme decks and save them to CSV files.

This script creates all the card decks used in the Modernisme game
and exports them to CSV files for reference and validation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pytts import Deck
from examples.modernisme_game import ModernismeGame


def generate_all_decks():
    """Generate all decks and save them to CSV files."""
    print("Generating Modernisme card decks...\n")

    # Create a temporary game instance to access deck creation methods
    game = ModernismeGame()

    # Create output directory
    output_dir = "modernisme_decks"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save work deck
    print("1. Work Deck (112 cards)")
    works = game._create_work_deck()
    work_deck = Deck(works)
    work_deck.save_to_csv(f"{output_dir}/work_deck.csv")
    print(f"   Saved to {output_dir}/work_deck.csv")
    print(f"   Total cards: {len(works)}")

    # Generate and save artist deck
    print("\n2. Artist Deck (28 cards)")
    artists = game._create_artist_deck()
    artist_deck = Deck(artists)
    artist_deck.save_to_csv(f"{output_dir}/artist_deck.csv")
    print(f"   Saved to {output_dir}/artist_deck.csv")
    print(f"   Total cards: {len(artists)}")

    # Generate and save moda tema cards
    print("\n3. Moda Tema Cards (4 cards)")
    moda_tema = game._create_moda_tema_cards()
    moda_tema_deck = Deck(moda_tema)
    moda_tema_deck.save_to_csv(f"{output_dir}/moda_tema_cards.csv")
    print(f"   Saved to {output_dir}/moda_tema_cards.csv")
    print(f"   Total cards: {len(moda_tema)}")

    # Generate and save moda conjunto cards
    print("\n4. Moda Conjunto Cards (4 cards)")
    moda_conjunto = game._create_moda_conjunto_cards()
    moda_conjunto_deck = Deck(moda_conjunto)
    moda_conjunto_deck.save_to_csv(f"{output_dir}/moda_conjunto_cards.csv")
    print(f"   Saved to {output_dir}/moda_conjunto_cards.csv")
    print(f"   Total cards: {len(moda_conjunto)}")

    # Generate and save encargo cards
    print("\n5. Encargo Cards (20 cards)")
    encargo = game._create_encargo_cards()
    encargo_deck = Deck(encargo)
    encargo_deck.save_to_csv(f"{output_dir}/encargo_cards.csv")
    print(f"   Saved to {output_dir}/encargo_cards.csv")
    print(f"   Total cards: {len(encargo)}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Work cards: {len(works)}")
    print(f"Artist cards: {len(artists)}")
    print(f"Moda Tema cards: {len(moda_tema)}")
    print(f"Moda Conjunto cards: {len(moda_conjunto)}")
    print(f"Encargo cards: {len(encargo)}")
    print(f"{'='*60}")
    print(f"\nAll decks saved to '{output_dir}/' directory")


if __name__ == "__main__":
    generate_all_decks()
