"""
Run large-scale Modernisme game simulations and analyze results.

This script runs thousands of games, saves logs and data to files,
then analyzes the results to determine strategy performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modernisme_game import play_modernisme_game
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List


def ensure_logs_directory():
    """Create the modernisme_logs directory if it doesn't exist."""
    logs_dir = Path(__file__).parent / "modernisme_logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def run_single_simulation(game_num: int, logs_dir: Path) -> str:
    """
    Run a single game and save logs and data.

    Args:
        game_num: The game number (for unique naming)
        logs_dir: Directory to save logs

    Returns:
        Path to the CSV file created
    """
    # Generate unique timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"game_{game_num:06d}_{timestamp}"

    log_file_path = logs_dir / f"{base_name}.log"
    csv_file_path = logs_dir / f"{base_name}.csv"

    # Run game with log file
    with open(log_file_path, 'w') as log_file:
        game = play_modernisme_game(log_file=log_file, num_players=4)

    # Save game data to CSV
    game_data = game.get_game_data()

    # Write CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=game_data.keys())
        writer.writeheader()
        writer.writerow(game_data)

    return str(csv_file_path)


def run_simulations(num_games: int = 100000):
    """
    Run multiple game simulations.

    Args:
        num_games: Number of games to simulate
    """
    print(f"Starting {num_games:,} game simulations...")
    print(f"Timestamp: {datetime.now()}")

    logs_dir = ensure_logs_directory()
    print(f"Saving logs to: {logs_dir}")

    # Run simulations
    csv_files = []
    for i in range(num_games):
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i + 1:,} / {num_games:,} games ({(i + 1) / num_games * 100:.1f}%)")

        csv_file = run_single_simulation(i + 1, logs_dir)
        csv_files.append(csv_file)

    print(f"\nCompleted {num_games:,} simulations!")
    print(f"All logs saved to: {logs_dir}")

    return csv_files


def analyze_results(logs_dir: Path):
    """
    Analyze all CSV files and compute statistics.

    Args:
        logs_dir: Directory containing CSV files
    """
    print("\n" + "=" * 70)
    print("ANALYZING SIMULATION RESULTS")
    print("=" * 70)

    # Read all CSV files
    csv_files = list(logs_dir.glob("game_*.csv"))
    print(f"\nFound {len(csv_files):,} game result files")

    if not csv_files:
        print("No game results found!")
        return

    # Load all data into a DataFrame
    print("Loading data...")
    df_list = []
    for csv_file in csv_files:
        df_list.append(pd.read_csv(csv_file))

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df):,} games")

    # Strategy win counts
    print("\n" + "=" * 70)
    print("STRATEGY WIN COUNTS")
    print("=" * 70)

    strategy_wins = df['winner_strategy'].value_counts().sort_index()
    total_games = len(df)

    print(f"\n{'Strategy':<30} {'Wins':>10} {'Win Rate':>10}")
    print("-" * 70)
    for strategy, wins in strategy_wins.items():
        win_rate = wins / total_games * 100
        print(f"{strategy:<30} {wins:>10,} {win_rate:>9.2f}%")

    # Win counts by starting position for each strategy
    print("\n" + "=" * 70)
    print("STRATEGY WINS BY STARTING POSITION")
    print("=" * 70)

    # Create a mapping of which strategy was in which position for each game
    strategies_by_position = {}
    for pos in range(1, 5):
        strategy_col = f'player_{pos}_strategy'

        # For each game, check if winner_position matches this position
        wins_by_strategy = {}
        for strategy in df[strategy_col].unique():
            # Count games where this strategy was in this position AND won
            mask = (df[strategy_col] == strategy) & (df['winner_position'] == pos)
            wins = mask.sum()

            # Total games where this strategy was in this position
            total = (df[strategy_col] == strategy).sum()

            if total > 0:
                if strategy not in wins_by_strategy:
                    wins_by_strategy[strategy] = {'total': 0, 'wins': [0, 0, 0, 0]}
                wins_by_strategy[strategy]['total'] += total
                wins_by_strategy[strategy]['wins'][pos - 1] = wins

        strategies_by_position[pos] = wins_by_strategy

    # Consolidate and display
    all_strategies = set()
    for pos_data in strategies_by_position.values():
        all_strategies.update(pos_data.keys())

    print(f"\n{'Strategy':<30} {'Position 1':>12} {'Position 2':>12} {'Position 3':>12} {'Position 4':>12} {'Total Games':>12}")
    print("-" * 110)

    for strategy in sorted(all_strategies):
        wins = [0, 0, 0, 0]
        games_played = [0, 0, 0, 0]

        for pos in range(1, 5):
            if strategy in strategies_by_position[pos]:
                wins[pos - 1] = strategies_by_position[pos][strategy]['wins'][pos - 1]
                games_played[pos - 1] = strategies_by_position[pos][strategy]['total']

        # Format: wins/games (win%)
        pos_strs = []
        for i in range(4):
            if games_played[i] > 0:
                win_pct = wins[i] / games_played[i] * 100
                pos_strs.append(f"{wins[i]}/{games_played[i]} ({win_pct:.1f}%)")
            else:
                pos_strs.append("0/0 (0.0%)")

        total_games = sum(games_played)
        print(f"{strategy:<30} {pos_strs[0]:>12} {pos_strs[1]:>12} {pos_strs[2]:>12} {pos_strs[3]:>12} {total_games:>12,}")

    # VP statistics by strategy
    print("\n" + "=" * 70)
    print("VICTORY POINTS STATISTICS BY STRATEGY")
    print("=" * 70)

    # Gather all scores for each strategy
    strategy_scores = {}
    for pos in range(1, 5):
        strategy_col = f'player_{pos}_strategy'
        score_col = f'player_{pos}_score'

        for idx, row in df.iterrows():
            strategy = row[strategy_col]
            score = row[score_col]

            if strategy not in strategy_scores:
                strategy_scores[strategy] = []
            strategy_scores[strategy].append(score)

    print(f"\n{'Strategy':<30} {'Avg VP':>10} {'Std Dev':>10} {'Min VP':>10} {'Max VP':>10} {'Games':>10}")
    print("-" * 90)

    for strategy in sorted(strategy_scores.keys()):
        scores = strategy_scores[strategy]
        avg_vp = np.mean(scores)
        std_vp = np.std(scores)
        min_vp = np.min(scores)
        max_vp = np.max(scores)

        print(f"{strategy:<30} {avg_vp:>10.2f} {std_vp:>10.2f} {min_vp:>10} {max_vp:>10} {len(scores):>10,}")

    # Score difference statistics
    print("\n" + "=" * 70)
    print("SCORE DIFFERENCE STATISTICS")
    print("=" * 70)

    print(f"\nAverage score difference: {df['score_difference'].mean():.2f}")
    print(f"Std dev score difference: {df['score_difference'].std():.2f}")
    print(f"Minimum score difference: {df['score_difference'].min()}")
    print(f"Maximum score difference: {df['score_difference'].max()}")

    # Distribution of score differences
    print(f"\nScore difference distribution:")
    diff_bins = [0, 5, 10, 15, 20, 25, 30, float('inf')]
    diff_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30+']

    for i in range(len(diff_labels)):
        if i < len(diff_bins) - 1:
            count = ((df['score_difference'] >= diff_bins[i]) & (df['score_difference'] < diff_bins[i + 1])).sum()
            pct = count / len(df) * 100
            print(f"  {diff_labels[i]:>8}: {count:>8,} games ({pct:>5.2f}%)")

    # Save summary statistics
    summary_file = logs_dir / "simulation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("MODERNISME SIMULATION SUMMARY\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total games: {total_games:,}\n")
        f.write(f"Analysis date: {datetime.now()}\n\n")

        f.write("STRATEGY WIN COUNTS\n")
        f.write("-" * 70 + "\n")
        for strategy, wins in strategy_wins.items():
            win_rate = wins / total_games * 100
            f.write(f"{strategy:<30} {wins:>10,} ({win_rate:>5.2f}%)\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\nSummary saved to: {summary_file}")


def main():
    """Main entry point for simulations."""
    # You can adjust the number of simulations here
    # Start with a smaller number for testing
    num_simulations = 100000

    if len(sys.argv) > 1:
        try:
            num_simulations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python run_simulations.py [num_games]")
            return

    print("=" * 70)
    print("MODERNISME SIMULATION RUNNER")
    print("=" * 70)

    # Run simulations
    logs_dir = ensure_logs_directory()
    run_simulations(num_simulations)

    # Analyze results
    analyze_results(logs_dir)

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
