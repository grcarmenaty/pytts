"""
Run large-scale Modernisme game simulations and analyze results.

This script runs thousands of games in parallel using multiprocessing,
saves logs and data to files, then analyzes the results to determine
strategy performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modernisme_game import play_modernisme_game
from generate_report import generate_pdf_report
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List
from multiprocessing import Pool, cpu_count
import time


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


def run_simulations(num_games: int = 100000, num_processes: int = None):
    """
    Run multiple game simulations using multiprocessing.

    Args:
        num_games: Number of games to simulate
        num_processes: Number of parallel processes (default: CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()

    print(f"Starting {num_games:,} game simulations...")
    print(f"Using {num_processes} parallel processes")
    print(f"Timestamp: {datetime.now()}")

    logs_dir = ensure_logs_directory()
    print(f"Saving logs to: {logs_dir}")

    # Prepare arguments for parallel execution
    start_time = time.time()
    args = [(i + 1, logs_dir) for i in range(num_games)]

    # Run simulations in parallel with progress tracking
    csv_files = []
    print()  # Newline for progress updates
    with Pool(processes=num_processes) as pool:
        # Use imap for progress tracking
        completed = 0
        for result in pool.starmap(run_single_simulation, args, chunksize=10):
            csv_files.append(result)
            completed += 1

            # Progress updates
            if completed % 100 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (num_games - completed) / rate if rate > 0 else 0
                print(f"  Progress: {completed:,} / {num_games:,} games "
                      f"({completed / num_games * 100:.1f}%) - "
                      f"Rate: {rate:.1f} games/sec - "
                      f"ETA: {remaining/60:.1f} min")

    elapsed_time = time.time() - start_time
    print(f"\nCompleted {num_games:,} simulations in {elapsed_time/60:.1f} minutes!")
    print(f"Average: {num_games/elapsed_time:.1f} games/second")
    print(f"All logs saved to: {logs_dir}")

    return csv_files


def aggregate_csvs(logs_dir: Path) -> str:
    """
    Aggregate all individual game CSV files into one master CSV.

    Args:
        logs_dir: Directory containing CSV files

    Returns:
        Path to the aggregated CSV file
    """
    print("\n" + "=" * 70)
    print("AGGREGATING CSV FILES")
    print("=" * 70)

    # Find all game CSV files
    csv_files = sorted(logs_dir.glob("game_*.csv"))
    print(f"\nFound {len(csv_files):,} game result files")

    if not csv_files:
        print("No game results found!")
        return None

    # Load and concatenate all CSVs
    print("Loading and combining all CSV files...")
    start_time = time.time()

    df_list = []
    skipped = 0
    for i, csv_file in enumerate(csv_files):
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1:,} / {len(csv_files):,} files...")
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                df_list.append(df)
            else:
                skipped += 1
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            skipped += 1
            continue

    if skipped > 0:
        print(f"  Skipped {skipped:,} empty or corrupted CSV files")

    if not df_list:
        print("ERROR: No valid CSV files found after filtering!")
        return None

    df = pd.concat(df_list, ignore_index=True)

    # Save aggregated CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregated_file = logs_dir / f"all_games_{timestamp}.csv"

    print(f"Saving aggregated CSV with {len(df):,} games...")
    df.to_csv(aggregated_file, index=False)

    elapsed = time.time() - start_time
    file_size_mb = aggregated_file.stat().st_size / (1024 * 1024)

    print(f"✓ Aggregated CSV saved: {aggregated_file}")
    print(f"  - {len(df):,} games")
    print(f"  - {len(df.columns)} columns")
    print(f"  - {file_size_mb:.1f} MB")
    print(f"  - Completed in {elapsed:.1f} seconds")

    return str(aggregated_file)


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
    batch_size = 10000
    for i in range(0, len(csv_files), batch_size):
        batch = csv_files[i:i+batch_size]
        if (i + len(batch)) % batch_size == 0 or i + len(batch) == len(csv_files):
            print(f"  Loading batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size}...")
        for csv_file in batch:
            df_list.append(pd.read_csv(csv_file))

    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df):,} games with {len(df.columns)} columns")

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
        strategy_col = f'p{pos}_strategy'

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
        strategy_col = f'p{pos}_strategy'
        score_col = f'p{pos}_final_score'

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
    num_processes = None  # Default to CPU count

    if len(sys.argv) > 1:
        try:
            num_simulations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python run_simulations.py [num_games] [num_processes]")
            return

    if len(sys.argv) > 2:
        try:
            num_processes = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of processes: {sys.argv[2]}")
            print("Usage: python run_simulations.py [num_games] [num_processes]")
            return

    print("=" * 70)
    print("MODERNISME SIMULATION RUNNER")
    print("=" * 70)

    # Run simulations
    logs_dir = ensure_logs_directory()
    run_simulations(num_simulations, num_processes)

    # Aggregate all CSVs into one
    aggregated_csv = aggregate_csvs(logs_dir)

    # Analyze results
    analyze_results(logs_dir)

    # Generate PDF report
    if aggregated_csv:
        print("\n" + "=" * 70)
        print("GENERATING PDF REPORT")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = logs_dir / f"simulation_report_{timestamp}.pdf"

        try:
            generate_pdf_report(aggregated_csv, str(pdf_path))
            print(f"\n✓ PDF report generated successfully!")
            print(f"  Location: {pdf_path}")
        except Exception as e:
            print(f"\n✗ Error generating PDF report: {e}")
            print("  (Analysis results are still available in text format)")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
