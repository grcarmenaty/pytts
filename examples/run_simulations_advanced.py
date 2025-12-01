"""
Run large-scale Modernisme Advanced game simulations and analyze results.

This script runs thousands of advanced mode games in parallel using multiprocessing,
saves logs and data to files, then analyzes the results to determine strategy
performance in the advanced game mode with room tiles and advantage cards.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modernisme_advanced import play_modernisme_advanced_game
from generate_report import generate_pdf_report
from pytts.strategy import ALL_STRATEGIES
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm
import time


def ensure_logs_directory():
    """Create the modernisme_advanced_logs directory with timestamped subfolder."""
    base_logs_dir = Path(__file__).parent / "modernisme_advanced_logs"
    base_logs_dir.mkdir(exist_ok=True)

    # Create timestamped subfolder for this simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = base_logs_dir / f"run_{timestamp}"
    run_logs_dir.mkdir(exist_ok=True)

    return run_logs_dir


def run_single_simulation(game_num: int, logs_dir: Path, strategy_classes: Tuple = None) -> str:
    """
    Run a single advanced game and save logs and data.

    Args:
        game_num: The game number (for unique naming)
        logs_dir: Directory to save logs
        strategy_classes: Tuple of strategy classes for each player

    Returns:
        Path to the CSV file created
    """
    # Create subdirectory based on game number to avoid single-directory bottleneck
    subdir_num = (game_num - 1) // 1000
    subdir = logs_dir / f"subdir_{subdir_num:04d}"
    subdir.mkdir(exist_ok=True)

    # Generate unique timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"game_{game_num:06d}_{timestamp}"

    log_file_path = subdir / f"{base_name}.log"
    csv_file_path = subdir / f"{base_name}.csv"

    # Run advanced game with log file
    with open(log_file_path, 'w') as log_file:
        game = play_modernisme_advanced_game(log_file=log_file, num_players=4,
                                            strategy_classes=list(strategy_classes) if strategy_classes else None)

    # Save game data to CSV
    game_data = game.get_game_data()

    # Write CSV file
    with open(csv_file_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=game_data.keys())
        writer.writeheader()
        writer.writerow(game_data)

    return str(csv_file_path)


def run_simulation_wrapper(args_tuple):
    """Wrapper for imap_unordered to unpack arguments."""
    return run_single_simulation(*args_tuple)


def generate_strategy_combinations(num_players: int = 4) -> List[Tuple]:
    """
    Generate all possible strategy combinations for the given number of players.

    Args:
        num_players: Number of players in each game

    Returns:
        List of tuples, where each tuple contains strategy classes for one combination
    """
    # Generate all possible combinations of strategies
    combinations = list(product(ALL_STRATEGIES, repeat=num_players))
    return combinations


def run_simulations(num_games: int = 100000, num_processes: int = None):
    """
    Run multiple advanced game simulations using multiprocessing with balanced strategy combinations.

    Args:
        num_games: Target number of games to simulate
        num_processes: Number of parallel processes (default: CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()

    # Generate all strategy combinations
    strategy_combinations = generate_strategy_combinations(num_players=4)
    num_combinations = len(strategy_combinations)

    # Calculate games per combination
    games_per_combination = max(1, (num_games + num_combinations - 1) // num_combinations)
    actual_num_games = games_per_combination * num_combinations

    print(f"Starting balanced ADVANCED MODE simulation with {actual_num_games:,} games...")
    print(f"  - {num_combinations:,} unique strategy combinations")
    print(f"  - {games_per_combination:,} games per combination")
    print(f"Using {num_processes} parallel processes")
    print(f"Timestamp: {datetime.now()}")

    logs_dir = ensure_logs_directory()
    print(f"Saving logs to: {logs_dir}")

    # Prepare arguments for parallel execution
    start_time = time.time()
    args = []
    game_num = 1
    for _ in range(games_per_combination):
        for strategy_combo in strategy_combinations:
            args.append((game_num, logs_dir, strategy_combo))
            game_num += 1

    # Run simulations in parallel with tqdm progress bar
    csv_files = []
    print()  # Newline before progress bar

    chunksize = max(1, min(20, actual_num_games // (num_processes * 100)))

    with Pool(processes=num_processes) as pool:
        with tqdm(total=actual_num_games, desc="Running advanced simulations",
                  unit="game", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                  smoothing=0.1) as pbar:
            for result in pool.imap_unordered(run_simulation_wrapper, args, chunksize=chunksize):
                csv_files.append(result)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n✓ Completed {actual_num_games:,} advanced simulations in {elapsed_time/60:.1f} minutes!")
    print(f"  Average: {actual_num_games/elapsed_time:.1f} games/second")
    print(f"  All logs saved to: {logs_dir}")

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

    # Find all game CSV files in subdirectories
    csv_files = sorted(logs_dir.glob("*/game_*.csv"))
    print(f"\nFound {len(csv_files):,} game result files")

    if not csv_files:
        print("No game results found!")
        return None

    # Load and concatenate all CSVs
    print("Loading and combining all CSV files...")
    start_time = time.time()

    df_list = []
    skipped = 0
    combination_logs = {}

    for csv_file in tqdm(csv_files, desc="Loading CSVs", unit="file", ncols=100):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                df_list.append(df)

                # Extract strategy combination
                strategies = tuple(df.iloc[0][f'p{i}_strategy'] for i in range(1, 5))

                # Get corresponding log file
                log_file = csv_file.with_suffix('.log')
                if log_file.exists():
                    if strategies not in combination_logs:
                        combination_logs[strategies] = []
                    combination_logs[strategies].append(log_file)
            else:
                skipped += 1
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            skipped += 1
            continue

    if skipped > 0:
        print(f"\n  Skipped {skipped:,} empty or corrupted CSV files")

    if not df_list:
        print("ERROR: No valid CSV files found after filtering!")
        return None

    df = pd.concat(df_list, ignore_index=True)

    # Save aggregated CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = logs_dir.parent
    aggregated_file = parent_dir / f"all_games_advanced_{timestamp}.csv"

    print(f"Saving aggregated CSV with {len(df):,} games...")
    df.to_csv(aggregated_file, index=False)

    elapsed = time.time() - start_time
    file_size_mb = aggregated_file.stat().st_size / (1024 * 1024)

    print(f"✓ Aggregated CSV saved: {aggregated_file}")
    print(f"  - {len(df):,} games")
    print(f"  - {len(df.columns)} columns")
    print(f"  - {file_size_mb:.1f} MB")
    print(f"  - Completed in {elapsed:.1f} seconds")

    # Select one random log per strategy combination
    import random
    sample_logs = set()
    for strategies, log_list in combination_logs.items():
        if log_list:
            sample_logs.add(random.choice(log_list))

    print(f"\nKeeping {len(sample_logs):,} sample log files (one per strategy combination)")

    # Delete individual CSV and log files (except sample logs)
    print("Cleaning up individual files...")
    deleted_count = 0
    kept_logs_count = 0
    log_files = list(logs_dir.glob("*/game_*.log"))
    all_files_to_delete = csv_files + [log for log in log_files if log not in sample_logs]

    for file_path in tqdm(all_files_to_delete, desc="Deleting files", unit="file", ncols=100):
        try:
            file_path.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"\nWarning: Could not delete {file_path}: {e}")

    kept_logs_count = len(sample_logs)
    print(f"\n✓ Deleted {deleted_count:,} files ({len(csv_files):,} CSVs + {len(log_files) - kept_logs_count:,} logs)")
    print(f"✓ Kept {kept_logs_count:,} sample log files")

    # Move sample logs to parent folder
    if sample_logs:
        print("\nMoving sample logs to parent folder...")
        samples_dir = parent_dir / "sample_logs"
        samples_dir.mkdir(exist_ok=True)

        for log_file in tqdm(sample_logs, desc="Moving sample logs", unit="file", ncols=100):
            try:
                new_name = f"{timestamp}_{log_file.name}"
                new_path = samples_dir / new_name
                log_file.rename(new_path)
            except Exception as e:
                print(f"\nWarning: Could not move {log_file}: {e}")

        print(f"\n✓ Sample logs saved to: {samples_dir}")

    # Remove empty subdirectories
    print("\nCleaning up empty subdirectories...")
    subdirs = list(logs_dir.glob("subdir_*"))
    removed_subdirs = 0
    for subdir in subdirs:
        try:
            subdir.rmdir()
            removed_subdirs += 1
        except OSError:
            pass
    if removed_subdirs > 0:
        print(f"✓ Removed {removed_subdirs} empty subdirectories")

    # Remove the now-empty run folder
    try:
        logs_dir.rmdir()
        print(f"✓ Removed empty run folder: {logs_dir.name}")
    except Exception as e:
        print(f"Note: Could not remove run folder (may not be empty): {e}")

    return str(aggregated_file)


def analyze_results(csv_file_path: str):
    """
    Analyze aggregated CSV file and compute statistics for advanced mode.

    Args:
        csv_file_path: Path to aggregated CSV file
    """
    print("\n" + "=" * 70)
    print("ANALYZING ADVANCED MODE SIMULATION RESULTS")
    print("=" * 70)

    # Load aggregated CSV
    print(f"\nLoading aggregated data from: {Path(csv_file_path).name}")
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df):,} games with {len(df.columns)} columns")

    # Get parent directory for saving summary
    logs_dir = Path(csv_file_path).parent

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

    # Win counts by starting position
    print("\n" + "=" * 70)
    print("STRATEGY WINS BY STARTING POSITION")
    print("=" * 70)

    strategies_by_position = {}
    for pos in range(1, 5):
        strategy_col = f'p{pos}_strategy'

        wins_by_strategy = {}
        for strategy in df[strategy_col].unique():
            mask = (df[strategy_col] == strategy) & (df['winner_position'] == pos)
            wins = mask.sum()

            total = (df[strategy_col] == strategy).sum()

            if total > 0:
                if strategy not in wins_by_strategy:
                    wins_by_strategy[strategy] = {'total': 0, 'wins': [0, 0, 0, 0]}
                wins_by_strategy[strategy]['total'] += total
                wins_by_strategy[strategy]['wins'][pos - 1] = wins

        strategies_by_position[pos] = wins_by_strategy

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

        pos_strs = []
        for i in range(4):
            if games_played[i] > 0:
                win_pct = wins[i] / games_played[i] * 100
                pos_strs.append(f"{wins[i]}/{games_played[i]} ({win_pct:.1f}%)")
            else:
                pos_strs.append("0/0 (0.0%)")

        total_games_strat = sum(games_played)
        print(f"{strategy:<30} {pos_strs[0]:>12} {pos_strs[1]:>12} {pos_strs[2]:>12} {pos_strs[3]:>12} {total_games_strat:>12,}")

    # VP statistics
    print("\n" + "=" * 70)
    print("VICTORY POINTS STATISTICS BY STRATEGY")
    print("=" * 70)

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

    # Advanced mode specific stats - Room tiles
    print("\n" + "=" * 70)
    print("ADVANCED MODE: ROOM TILE STATISTICS")
    print("=" * 70)

    room_tile_stats = {}
    for pos in range(1, 5):
        strategy_col = f'p{pos}_strategy'
        room_tiles_col = f'p{pos}_room_tiles'

        for idx, row in df.iterrows():
            strategy = row[strategy_col]
            if room_tiles_col in row:
                room_tiles = row[room_tiles_col]

                if strategy not in room_tile_stats:
                    room_tile_stats[strategy] = []
                room_tile_stats[strategy].append(room_tiles)

    if room_tile_stats:
        print(f"\n{'Strategy':<30} {'Avg Tiles':>12} {'Min':>8} {'Max':>8}")
        print("-" * 70)

        for strategy in sorted(room_tile_stats.keys()):
            tiles = room_tile_stats[strategy]
            avg_tiles = np.mean(tiles)
            min_tiles = np.min(tiles)
            max_tiles = np.max(tiles)

            print(f"{strategy:<30} {avg_tiles:>12.2f} {min_tiles:>8} {max_tiles:>8}")

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

    # Head-to-head matchup analysis
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD WIN RATES")
    print("=" * 70)

    strategy_usage = {}
    for pos in range(1, 5):
        strategy_col = f'p{pos}_strategy'
        for strategy in df[strategy_col]:
            if strategy not in strategy_usage:
                strategy_usage[strategy] = 0
            strategy_usage[strategy] += 1

    all_strat_names = sorted(strategy_usage.keys())
    h2h_matrix = {s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names} for s1 in all_strat_names}

    for idx, row in df.iterrows():
        winner_strategy = row['winner_strategy']
        winner_pos = row['winner_position']

        strategies = [row[f'p{pos}_strategy'] for pos in range(1, 5)]

        for pos in range(1, 5):
            if pos != winner_pos:
                opponent = strategies[pos - 1]
                h2h_matrix[winner_strategy][opponent]['wins'] += 1
                h2h_matrix[winner_strategy][opponent]['games'] += 1
                h2h_matrix[opponent][winner_strategy]['games'] += 1

    # Display matrix
    print(f"\n{'Strategy':<25}", end='')
    for s in all_strat_names:
        print(f" {s[:10]:>10}", end='')
    print()
    print("-" * (25 + 11 * len(all_strat_names)))

    for s1 in all_strat_names:
        print(f"{s1:<25}", end='')
        for s2 in all_strat_names:
            if s1 == s2:
                print(f" {'--':>10}", end='')
            else:
                games = h2h_matrix[s1][s2]['games']
                if games > 0:
                    wins = h2h_matrix[s1][s2]['wins']
                    win_rate = wins / games * 100
                    print(f" {win_rate:>9.1f}%", end='')
                else:
                    print(f" {'0.0%':>10}", end='')
        print()

    # Save summary statistics
    summary_file = logs_dir / "simulation_summary_advanced.txt"
    with open(summary_file, 'w') as f:
        f.write("MODERNISME ADVANCED MODE SIMULATION SUMMARY\n")
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
    """Main entry point for advanced mode simulations."""
    num_simulations = 100000
    num_processes = None

    if len(sys.argv) > 1:
        try:
            num_simulations = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number: {sys.argv[1]}")
            print("Usage: python run_simulations_advanced.py [num_games] [num_processes]")
            return

    if len(sys.argv) > 2:
        try:
            num_processes = int(sys.argv[2])
        except ValueError:
            print(f"Invalid number of processes: {sys.argv[2]}")
            print("Usage: python run_simulations_advanced.py [num_games] [num_processes]")
            return

    print("=" * 70)
    print("MODERNISME ADVANCED MODE SIMULATION RUNNER")
    print("=" * 70)

    # Run simulations
    logs_dir = ensure_logs_directory()
    run_simulations(num_simulations, num_processes)

    # Aggregate all CSVs
    aggregated_csv = aggregate_csvs(logs_dir)

    # Analyze results
    if aggregated_csv:
        analyze_results(aggregated_csv)

    # Generate PDF report
    if aggregated_csv:
        print("\n" + "=" * 70)
        print("GENERATING PDF REPORT")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_logs_dir = Path(aggregated_csv).parent
        pdf_path = parent_logs_dir / f"simulation_report_advanced_{timestamp}.pdf"

        try:
            generate_pdf_report(aggregated_csv, str(pdf_path))
            print(f"\n✓ PDF report generated successfully!")
            print(f"  Location: {pdf_path}")
        except Exception as e:
            print(f"\n✗ Error generating PDF report: {e}")
            print("  (Analysis results are still available in text format)")

    print("\n" + "=" * 70)
    print("ADVANCED MODE SIMULATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
