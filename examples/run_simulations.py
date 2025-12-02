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
    """Create the modernisme_logs directory with timestamped subfolder."""
    base_logs_dir = Path(__file__).parent / "modernisme_logs"
    base_logs_dir.mkdir(exist_ok=True)

    # Create timestamped subfolder for this simulation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logs_dir = base_logs_dir / f"run_{timestamp}"
    run_logs_dir.mkdir(exist_ok=True)

    return run_logs_dir


def run_single_simulation(game_num: int, logs_dir: Path, num_players: int = 4, strategy_classes: Tuple = None) -> str:
    """
    Run a single game and save logs and data.

    Args:
        game_num: The game number (for unique naming)
        logs_dir: Directory to save logs
        num_players: Number of players in the game
        strategy_classes: Tuple of strategy classes for each player

    Returns:
        Path to the CSV file created
    """
    # Create subdirectory based on game number to avoid single-directory bottleneck
    # e.g., game 1-1000 -> subdir_0000/, game 1001-2000 -> subdir_0001/
    subdir_num = (game_num - 1) // 1000
    subdir = logs_dir / f"subdir_{subdir_num:04d}"
    subdir.mkdir(exist_ok=True)

    # Generate unique timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"game_{game_num:06d}_{num_players}p_{timestamp}"

    log_file_path = subdir / f"{base_name}.log"
    csv_file_path = subdir / f"{base_name}.csv"

    # Run game with log file
    with open(log_file_path, 'w') as log_file:
        game = play_modernisme_game(log_file=log_file, num_players=num_players,
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


def generate_strategy_combinations() -> List[Tuple]:
    """
    Generate all possible strategy combinations for 2, 3, and 4 players.

    Returns:
        List of tuples, where each tuple contains (num_players, strategy_classes)
    """
    combinations = []

    # Generate combinations for 2, 3, and 4 players
    for num_players in [2, 3, 4]:
        player_combos = list(product(ALL_STRATEGIES, repeat=num_players))
        # Add num_players to each combination tuple
        combinations.extend([(num_players, combo) for combo in player_combos])

    return combinations


def run_simulations(num_games: int = 100000, num_processes: int = None):
    """
    Run multiple game simulations using multiprocessing with balanced strategy combinations.

    Each possible combination of strategies (for 2, 3, and 4 players) is run an equal number of times.

    Args:
        num_games: Target number of games to simulate (will be adjusted to ensure even distribution)
        num_processes: Number of parallel processes (default: CPU count)
    """
    if num_processes is None:
        num_processes = cpu_count()

    # Generate all strategy combinations for 2, 3, and 4 players
    strategy_combinations = generate_strategy_combinations()
    num_combinations = len(strategy_combinations)

    # Calculate games per combination to reach or exceed target
    games_per_combination = max(1, (num_games + num_combinations - 1) // num_combinations)
    actual_num_games = games_per_combination * num_combinations

    # Count by player count
    player_counts = {2: 0, 3: 0, 4: 0}
    for num_players, _ in strategy_combinations:
        player_counts[num_players] += games_per_combination

    print(f"Starting balanced simulation with {actual_num_games:,} games...")
    print(f"  - {num_combinations:,} unique strategy combinations")
    print(f"  - {games_per_combination:,} games per combination")
    print(f"  - 2-player games: {player_counts[2]:,}")
    print(f"  - 3-player games: {player_counts[3]:,}")
    print(f"  - 4-player games: {player_counts[4]:,}")
    print(f"Using {num_processes} parallel processes")
    print(f"Timestamp: {datetime.now()}")

    logs_dir = ensure_logs_directory()
    print(f"Saving logs to: {logs_dir}")

    # Prepare arguments for parallel execution
    # Each combination is run games_per_combination times
    start_time = time.time()
    args = []
    game_num = 1
    for _ in range(games_per_combination):
        for num_players, strategy_combo in strategy_combinations:
            args.append((game_num, logs_dir, num_players, strategy_combo))
            game_num += 1

    # Run simulations in parallel with tqdm progress bar
    csv_files = []
    print()  # Newline before progress bar

    # Use larger chunksize to reduce task distribution overhead
    # Progress bar updates every ~10 games instead of every game
    chunksize = max(1, min(20, actual_num_games // (num_processes * 100)))

    with Pool(processes=num_processes) as pool:
        with tqdm(total=actual_num_games, desc="Running simulations",
                  unit="game", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                  smoothing=0.1) as pbar:
            for result in pool.imap_unordered(run_simulation_wrapper, args, chunksize=chunksize):
                csv_files.append(result)
                pbar.update(1)

    elapsed_time = time.time() - start_time
    print(f"\n✓ Completed {actual_num_games:,} simulations in {elapsed_time/60:.1f} minutes!")
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
    # Track strategy combinations and their corresponding log files for sampling
    combination_logs = {}  # combination tuple -> list of log file paths

    for csv_file in tqdm(csv_files, desc="Loading CSVs", unit="file", ncols=100):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                df_list.append(df)

                # Extract strategy combination - detect number of players
                # Count how many p#_strategy columns exist
                num_players = 0
                for i in range(1, 5):
                    if f'p{i}_strategy' in df.columns:
                        num_players = i
                    else:
                        break

                strategies = tuple(df.iloc[0][f'p{i}_strategy'] for i in range(1, num_players + 1))

                # Get corresponding log file (same base name)
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

    # Save aggregated CSV to parent modernisme_logs folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = logs_dir.parent
    aggregated_file = parent_dir / f"all_games_{timestamp}.csv"

    print(f"Saving aggregated CSV with {len(df):,} games...")
    df.to_csv(aggregated_file, index=False)

    elapsed = time.time() - start_time
    file_size_mb = aggregated_file.stat().st_size / (1024 * 1024)

    print(f"✓ Aggregated CSV saved: {aggregated_file}")
    print(f"  - {len(df):,} games")
    print(f"  - {len(df.columns)} columns")
    print(f"  - {file_size_mb:.1f} MB")
    print(f"  - Completed in {elapsed:.1f} seconds")

    # Select one random log per strategy combination to keep
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

    # Move sample logs to parent folder with descriptive naming
    if sample_logs:
        print("\nMoving sample logs to parent folder...")
        samples_dir = parent_dir / "sample_logs"
        samples_dir.mkdir(exist_ok=True)

        for log_file in tqdm(sample_logs, desc="Moving sample logs", unit="file", ncols=100):
            try:
                # Create a descriptive name based on the original filename
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
            pass  # Not empty, skip
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
    Analyze aggregated CSV file and compute statistics.

    Args:
        csv_file_path: Path to aggregated CSV file
    """
    print("\n" + "=" * 70)
    print("ANALYZING SIMULATION RESULTS")
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

        # Skip if column doesn't exist
        if strategy_col not in df.columns or score_col not in df.columns:
            continue

        for idx, row in df.iterrows():
            strategy = row[strategy_col]
            score = row[score_col]

            # Skip NaN values (from games with fewer players)
            if pd.isna(strategy) or pd.isna(score):
                continue

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

    # Strategy usage frequency
    print("\n" + "=" * 70)
    print("STRATEGY USAGE FREQUENCY")
    print("=" * 70)

    strategy_usage = {}
    strategy_position_usage = {pos: {} for pos in range(1, 5)}

    for pos in range(1, 5):
        strategy_col = f'p{pos}_strategy'
        if strategy_col not in df.columns:
            continue
        for strategy in df[strategy_col]:
            # Skip NaN values
            if pd.isna(strategy):
                continue
            # Overall count
            if strategy not in strategy_usage:
                strategy_usage[strategy] = 0
            strategy_usage[strategy] += 1

            # Position-specific count
            if strategy not in strategy_position_usage[pos]:
                strategy_position_usage[pos][strategy] = 0
            strategy_position_usage[pos][strategy] += 1

    total_plays = sum(strategy_usage.values())
    print(f"\n{'Strategy':<30} {'Total Plays':>12} {'% of Games':>12} {'Pos 1':>8} {'Pos 2':>8} {'Pos 3':>8} {'Pos 4':>8}")
    print("-" * 110)

    for strategy in sorted(strategy_usage.keys()):
        count = strategy_usage[strategy]
        pct = count / total_plays * 100
        pos_counts = [strategy_position_usage[pos].get(strategy, 0) for pos in range(1, 5)]
        print(f"{strategy:<30} {count:>12,} {pct:>11.2f}% {pos_counts[0]:>8,} {pos_counts[1]:>8,} {pos_counts[2]:>8,} {pos_counts[3]:>8,}")

    # Head-to-head matchup analysis
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD WIN RATES")
    print("=" * 70)
    print("\nWhen Strategy A plays in ANY position against Strategy B in ANY position,")
    print("what is Strategy A's win rate?")

    # Build head-to-head matrix
    all_strat_names = sorted(strategy_usage.keys())
    h2h_matrix = {s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names} for s1 in all_strat_names}

    # For each game, record all head-to-head matchups
    for idx, row in df.iterrows():
        winner_strategy = row['winner_strategy']
        winner_pos = row['winner_position']

        # Skip if winner strategy is NaN
        if pd.isna(winner_strategy):
            continue

        # Collect strategies, skipping NaN values
        strategies = []
        for pos in range(1, 5):
            col = f'p{pos}_strategy'
            if col in df.columns and not pd.isna(row[col]):
                strategies.append((pos, row[col]))

        # Record win for winner against all opponents
        for pos, opponent in strategies:
            if pos != winner_pos:
                h2h_matrix[winner_strategy][opponent]['wins'] += 1
                h2h_matrix[winner_strategy][opponent]['games'] += 1
                # Also record loss for opponent
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

    # Matchup performance by opponent combination
    print("\n" + "=" * 70)
    print("MATCHUP PERFORMANCE MATRIX")
    print("=" * 70)
    print("\nFor each strategy, showing performance against different opponent combinations:")
    print("(Top 10 best and worst matchups for each strategy)")

    matchup_performance = {}

    for idx, row in df.iterrows():
        winner_strategy = row['winner_strategy']
        winner_pos = row['winner_position']

        # Get all strategies in this game
        strategies = tuple(row[f'p{pos}_strategy'] for pos in range(1, 5))

        # Record performance for each strategy in this game
        for pos in range(1, 5):
            strategy = strategies[pos - 1]
            # Get opponents (all strategies except current position)
            opponents = tuple(strategies[i] for i in range(4) if i != pos - 1)

            if strategy not in matchup_performance:
                matchup_performance[strategy] = {}
            if opponents not in matchup_performance[strategy]:
                matchup_performance[strategy][opponents] = {'wins': 0, 'games': 0}

            matchup_performance[strategy][opponents]['games'] += 1
            if pos == winner_pos:
                matchup_performance[strategy][opponents]['wins'] += 1

    # Display top/bottom matchups for each strategy
    for strategy in sorted(matchup_performance.keys()):
        matchups = matchup_performance[strategy]

        # Calculate win rates
        matchup_winrates = []
        for opponents, stats in matchups.items():
            if stats['games'] >= 1:  # At least 1 game for balanced simulations
                win_rate = stats['wins'] / stats['games'] * 100
                matchup_winrates.append((opponents, stats['wins'], stats['games'], win_rate))

        # Sort by win rate
        matchup_winrates.sort(key=lambda x: x[3], reverse=True)

        print(f"\n{strategy}:")
        print(f"  Best matchups:")
        for opponents, wins, games, win_rate in matchup_winrates[:5]:
            opp_str = ' / '.join([o[:15] for o in opponents])
            print(f"    vs [{opp_str}]: {wins}/{games} ({win_rate:.1f}%)")

        print(f"  Worst matchups:")
        for opponents, wins, games, win_rate in matchup_winrates[-5:]:
            opp_str = ' / '.join([o[:15] for o in opponents])
            print(f"    vs [{opp_str}]: {wins}/{games} ({win_rate:.1f}%)")

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

    # Analyze results from aggregated CSV
    if aggregated_csv:
        analyze_results(aggregated_csv)

    # Generate PDF report
    if aggregated_csv:
        print("\n" + "=" * 70)
        print("GENERATING PDF REPORT")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save PDF to parent modernisme_logs folder (not the run subfolder which is deleted)
        parent_logs_dir = Path(aggregated_csv).parent
        pdf_path = parent_logs_dir / f"simulation_report_{timestamp}.pdf"

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
