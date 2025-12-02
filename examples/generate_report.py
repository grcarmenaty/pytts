"""
Generate comprehensive PDF report from Modernisme simulation data.

This module creates a professional PDF report with:
- Executive summary
- Strategy performance analysis
- Statistical visualizations
- Position-based analysis
- Turn-by-turn insights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, List, Tuple


# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


class SimulationReport:
    """Generate comprehensive PDF report from simulation data."""

    def __init__(self, df: pd.DataFrame, output_path: str):
        """
        Initialize report generator.

        Args:
            df: DataFrame containing all game results
            output_path: Path to save PDF report
        """
        self.df = df
        self.output_path = output_path
        self.temp_dir = tempfile.mkdtemp()
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.story = []

    def _setup_custom_styles(self):
        """Create custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#1f4788'),
            spaceAfter=12,
            spaceBefore=12
        ))
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#2c5f2d'),
            leftIndent=20,
            bulletIndent=10
        ))

    def generate(self):
        """Generate the complete PDF report."""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )

        # Build report sections
        self._add_title_page()
        self._add_executive_summary()
        self._add_player_count_analysis()
        self._add_works_and_vp_statistics()
        self._add_advanced_mode_statistics()
        self._add_strategy_matchups_by_player_count()
        self._add_strategy_analysis()
        self._add_head_to_head_analysis()
        self._add_matchup_analysis()
        self._add_position_analysis()
        self._add_vp_analysis()
        self._add_key_insights()

        # Build PDF
        doc.build(self.story)
        print(f"✓ PDF report saved: {self.output_path}")

    def _add_title_page(self):
        """Add title page."""
        self.story.append(Spacer(1, 2*inch))

        title = Paragraph("Modernisme Strategy Simulation", self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))

        subtitle = Paragraph(
            f"Comprehensive Analysis Report",
            self.styles['Heading2']
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        date_text = Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            self.styles['Normal']
        )
        self.story.append(date_text)
        self.story.append(Spacer(1, 0.3*inch))

        # Summary stats box
        total_games = len(self.df)
        data = [
            ["Total Games Simulated", f"{total_games:,}"],
            ["Strategies Tested", f"{self.df['winner_strategy'].nunique()}"],
            ["Total Players", f"{total_games * 4:,}"],
            ["Data Points Analyzed", f"{len(self.df.columns) * total_games:,}"],
        ]

        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        self.story.append(table)
        self.story.append(PageBreak())

    def _add_executive_summary(self):
        """Add executive summary section."""
        self.story.append(Paragraph("Executive Summary", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        # Winner statistics
        winner_stats = self.df['winner_strategy'].value_counts()
        best_strategy = winner_stats.index[0]
        best_wins = winner_stats.iloc[0]
        best_rate = (best_wins / len(self.df)) * 100

        summary_text = f"""
        This report analyzes {len(self.df):,} simulated games of Modernisme, examining the performance
        of {self.df['winner_strategy'].nunique()} different AI strategies. The simulation provides
        comprehensive insights into strategy effectiveness, positional advantages, and scoring patterns.
        <br/><br/>
        <b>Top Performing Strategy:</b> {best_strategy} with {best_wins:,} wins ({best_rate:.1f}% win rate)
        <br/>
        <b>Average Game Score:</b> {self.df['avg_score'].mean():.1f} VP
        <br/>
        <b>Score Range:</b> {self.df['min_score'].min():.0f} - {self.df['max_score'].max():.0f} VP
        """

        self.story.append(Paragraph(summary_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.3*inch))

    def _add_player_count_analysis(self):
        """Add analysis broken down by player count."""
        self.story.append(PageBreak())

        title = Paragraph("Multi-Player Game Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes game outcomes across different player counts (2, 3, and 4 players).",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Determine player count for each game
        player_count_data = {}
        for idx, row in self.df.iterrows():
            # Count non-NaN strategy columns
            num_players = sum(1 for pos in range(1, 5)
                            if f'p{pos}_strategy' in self.df.columns
                            and not pd.isna(row[f'p{pos}_strategy']))

            if num_players not in player_count_data:
                player_count_data[num_players] = {'games': 0, 'strategies': {}}

            player_count_data[num_players]['games'] += 1

            # Track wins by strategy for this player count
            winner = row['winner_strategy']
            if not pd.isna(winner):
                if winner not in player_count_data[num_players]['strategies']:
                    player_count_data[num_players]['strategies'][winner] = 0
                player_count_data[num_players]['strategies'][winner] += 1

        # Create summary table
        table_data = [['Player Count', 'Total Games', '% of All Games', 'Avg Score', 'Avg Score Diff']]

        for player_count in sorted(player_count_data.keys()):
            data = player_count_data[player_count]
            total_games = data['games']
            pct = (total_games / len(self.df)) * 100

            # Calculate average scores for this player count
            mask = self.df.apply(lambda row: sum(1 for pos in range(1, 5)
                                                 if f'p{pos}_strategy' in self.df.columns
                                                 and not pd.isna(row[f'p{pos}_strategy'])) == player_count,
                                axis=1)
            subset = self.df[mask]

            avg_winner_score = subset['winner_score'].mean() if 'winner_score' in subset.columns else 0
            avg_score_diff = subset['score_difference'].mean() if 'score_difference' in subset.columns else 0

            table_data.append([
                f"{player_count} Players",
                f"{total_games:,}",
                f"{pct:.1f}%",
                f"{avg_winner_score:.1f}",
                f"{avg_score_diff:.1f}"
            ])

        table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

        # Strategy performance by player count
        subtitle = Paragraph("Strategy Performance by Player Count", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        # Create win rate table
        all_strategies = set()
        for data in player_count_data.values():
            all_strategies.update(data['strategies'].keys())

        strategy_table_data = [['Strategy', '2P Win Rate', '3P Win Rate', '4P Win Rate']]

        for strategy in sorted(all_strategies):
            row = [strategy]
            for pc in [2, 3, 4]:
                if pc in player_count_data:
                    wins = player_count_data[pc]['strategies'].get(strategy, 0)
                    games = player_count_data[pc]['games']
                    # In N-player games, each strategy has 1/N chance if equal
                    expected_wins = games / pc
                    win_rate = (wins / expected_wins * 100) if expected_wins > 0 else 0
                    row.append(f"{win_rate:.1f}%")
                else:
                    row.append("N/A")
            strategy_table_data.append(row)

        strat_table = Table(strategy_table_data, colWidths=[2.5*inch, 1.3*inch, 1.3*inch, 1.3*inch])
        strat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(strat_table)
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: Win rates are normalized by expected performance (100% = performs as expected for that player count).</i>",
            self.styles['BodyText']
        )
        self.story.append(note)

    def _add_works_and_vp_statistics(self):
        """Add comprehensive works played/discarded and VP spending statistics."""
        self.story.append(PageBreak())

        title = Paragraph("Works & VP Expenditure Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Comprehensive analysis of works played, cards discarded, and VP spending patterns across all strategies.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Aggregate data by strategy
        strategy_stats = {}

        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            if strategy_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                if pd.isna(strategy):
                    continue

                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        'total_works_played': [],
                        'total_cards_discarded': [],
                        'total_vp_earned': [],
                        'total_vp_spent': [],
                        'works_crafts': [],
                        'works_painting': [],
                        'works_sculpture': [],
                        'works_relic': [],
                        'works_nature': [],
                        'works_mythology': [],
                        'works_society': [],
                        'works_orientalism': [],
                        'works_no_theme': [],
                    }

                # Collect all stats
                if f'p{pos}_total_cards_played' in self.df.columns:
                    strategy_stats[strategy]['total_works_played'].append(row[f'p{pos}_total_cards_played'])
                if f'p{pos}_total_cards_discarded' in self.df.columns:
                    strategy_stats[strategy]['total_cards_discarded'].append(row[f'p{pos}_total_cards_discarded'])
                if f'p{pos}_total_vp_earned' in self.df.columns:
                    strategy_stats[strategy]['total_vp_earned'].append(row[f'p{pos}_total_vp_earned'])
                if f'p{pos}_total_vp_spent' in self.df.columns:
                    strategy_stats[strategy]['total_vp_spent'].append(row[f'p{pos}_total_vp_spent'])

                # Works by type
                for work_type in ['crafts', 'painting', 'sculpture', 'relic']:
                    col = f'p{pos}_works_{work_type}'
                    if col in self.df.columns:
                        strategy_stats[strategy][f'works_{work_type}'].append(row[col])

                # Works by theme
                for theme in ['nature', 'mythology', 'society', 'orientalism', 'no_theme']:
                    col = f'p{pos}_works_{theme}'
                    if col in self.df.columns:
                        strategy_stats[strategy][f'works_{theme}'].append(row[col])

        # Overall Summary Table
        subtitle = Paragraph("Overall Works & VP Summary by Strategy", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        table_data = [['Strategy', 'Avg Works\nPlayed', 'Avg Cards\nDiscarded', 'Avg VP\nEarned', 'Avg VP\nSpent']]

        for strategy in sorted(strategy_stats.keys()):
            stats = strategy_stats[strategy]
            avg_played = np.mean(stats['total_works_played']) if stats['total_works_played'] else 0
            avg_discarded = np.mean(stats['total_cards_discarded']) if stats['total_cards_discarded'] else 0
            avg_earned = np.mean(stats['total_vp_earned']) if stats['total_vp_earned'] else 0
            avg_spent = np.mean(stats['total_vp_spent']) if stats['total_vp_spent'] else 0

            table_data.append([
                strategy,
                f"{avg_played:.1f}",
                f"{avg_discarded:.1f}",
                f"{avg_earned:.1f}",
                f"{avg_spent:.1f}"
            ])

        table = Table(table_data, colWidths=[2.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

        # Works by Art Type breakdown
        subtitle = Paragraph("Works Played by Art Type", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        type_table_data = [['Strategy', 'Crafts', 'Painting', 'Sculpture', 'Relic']]

        for strategy in sorted(strategy_stats.keys()):
            stats = strategy_stats[strategy]
            row = [strategy]
            for work_type in ['crafts', 'painting', 'sculpture', 'relic']:
                avg = np.mean(stats[f'works_{work_type}']) if stats[f'works_{work_type}'] else 0
                row.append(f"{avg:.1f}")
            type_table_data.append(row)

        type_table = Table(type_table_data, colWidths=[2.2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        type_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(type_table)
        self.story.append(Spacer(1, 0.3*inch))

        # Works by Theme breakdown
        subtitle = Paragraph("Works Played by Theme", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        theme_table_data = [['Strategy', 'Nature', 'Mythology', 'Society', 'Orientalism', 'No Theme']]

        for strategy in sorted(strategy_stats.keys()):
            stats = strategy_stats[strategy]
            row = [strategy]
            for theme in ['nature', 'mythology', 'society', 'orientalism', 'no_theme']:
                avg = np.mean(stats[f'works_{theme}']) if stats[f'works_{theme}'] else 0
                row.append(f"{avg:.1f}")
            theme_table_data.append(row)

        theme_table = Table(theme_table_data, colWidths=[1.8*inch, 0.85*inch, 0.95*inch, 0.85*inch, 1*inch, 0.9*inch])
        theme_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(theme_table)

    def _add_advanced_mode_statistics(self):
        """Add advanced mode specific statistics (room tiles, artist/theme usage)."""
        # Check if this is advanced mode data
        has_room_tiles = 'p1_room_tiles' in self.df.columns
        if not has_room_tiles:
            return  # Skip this section for base game

        self.story.append(PageBreak())

        title = Paragraph("Advanced Mode Statistics", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        # Room Tile Acquisition Stats
        subtitle = Paragraph("Room Tile Acquisition", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Analysis of room tile acquisition patterns across all strategies.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate room tile stats by strategy
        tile_stats = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            tiles_col = f'p{pos}_room_tiles'

            if strategy_col not in self.df.columns or tiles_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                tiles = row[tiles_col]

                if pd.isna(strategy) or pd.isna(tiles):
                    continue

                if strategy not in tile_stats:
                    tile_stats[strategy] = []
                tile_stats[strategy].append(tiles)

        if tile_stats:
            table_data = [['Strategy', 'Avg Tiles', 'Min', 'Max', 'Std Dev']]

            for strategy in sorted(tile_stats.keys()):
                tiles = tile_stats[strategy]
                avg = np.mean(tiles)
                min_val = np.min(tiles)
                max_val = np.max(tiles)
                std = np.std(tiles)

                table_data.append([
                    strategy,
                    f"{avg:.2f}",
                    f"{min_val:.0f}",
                    f"{max_val:.0f}",
                    f"{std:.2f}"
                ])

            table = Table(table_data, colWidths=[2.2*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))

            self.story.append(table)
            self.story.append(Spacer(1, 0.3*inch))

        # Advantage Card Selection Statistics
        subtitle = Paragraph("Advantage Card Selection Statistics", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Analysis of which advantage cards are selected most frequently by each strategy.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Collect advantage card selections
        advantage_selections = {}
        total_selections_by_strategy = {}

        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            adv_col = f'p{pos}_advantage_cards'

            if strategy_col not in self.df.columns or adv_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                adv_cards = row[adv_col]

                if pd.isna(strategy) or pd.isna(adv_cards):
                    continue

                if strategy not in advantage_selections:
                    advantage_selections[strategy] = {}
                    total_selections_by_strategy[strategy] = 0

                # Parse comma-separated advantage cards
                cards = str(adv_cards).split(',')
                for card in cards:
                    card = card.strip()
                    if card:
                        if card not in advantage_selections[strategy]:
                            advantage_selections[strategy][card] = 0
                        advantage_selections[strategy][card] += 1
                        total_selections_by_strategy[strategy] += 1

        if advantage_selections:
            # Get all unique cards
            all_cards = set()
            for cards_dict in advantage_selections.values():
                all_cards.update(cards_dict.keys())

            # Create table showing selection frequency
            table_data = [['Advantage Card'] + sorted(advantage_selections.keys())]

            for card in sorted(all_cards):
                row = [card]
                for strategy in sorted(advantage_selections.keys()):
                    count = advantage_selections[strategy].get(card, 0)
                    total = total_selections_by_strategy[strategy]
                    pct = (count / total * 100) if total > 0 else 0
                    row.append(f"{count}\n({pct:.1f}%)")
                table_data.append(row)

            # Calculate column widths dynamically
            num_strategies = len(advantage_selections)
            card_col_width = 2.2*inch
            strategy_col_width = (6.5*inch - card_col_width) / num_strategies

            col_widths = [card_col_width] + [strategy_col_width] * num_strategies

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))

            self.story.append(table)
            self.story.append(Spacer(1, 0.3*inch))

        # Artist/Theme Usage (if available in logs)
        # Note: This would require parsing log files or adding columns to CSV
        # For now, add placeholder text
        subtitle = Paragraph("Artist and Theme Analysis", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        analysis_note = Paragraph(
            "Analysis of artist types and themes commissioned throughout games. "
            "This data reveals strategic preferences and meta-game trends.",
            self.styles['BodyText']
        )
        self.story.append(analysis_note)
        self.story.append(Spacer(1, 0.2*inch))

        # Check if we have works_placed columns
        works_data = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            works_col = f'p{pos}_works_placed'

            if strategy_col not in self.df.columns or works_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                works = row[works_col]

                if pd.isna(strategy) or pd.isna(works):
                    continue

                if strategy not in works_data:
                    works_data[strategy] = []
                works_data[strategy].append(works)

        if works_data:
            table_data = [['Strategy', 'Avg Works Placed', 'Min', 'Max']]

            for strategy in sorted(works_data.keys()):
                works = works_data[strategy]
                avg = np.mean(works)
                min_val = np.min(works)
                max_val = np.max(works)

                table_data.append([
                    strategy,
                    f"{avg:.2f}",
                    f"{min_val:.0f}",
                    f"{max_val:.0f}"
                ])

            table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ]))

            self.story.append(table)

    def _add_strategy_matchups_by_player_count(self):
        """Add detailed strategy matchup analysis broken down by player count."""
        self.story.append(PageBreak())

        title = Paragraph("Strategy Matchups by Player Count", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes how each strategy performs against specific combinations of opponents, "
            "broken down by 2-player, 3-player, and 4-player games. Win rates show performance against "
            "each unique opponent combination.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Analyze by player count
        for player_count in [2, 3, 4]:
            subtitle = Paragraph(f"{player_count}-Player Game Matchups", self.styles['Heading2'])
            self.story.append(subtitle)
            self.story.append(Spacer(1, 0.2*inch))

            # Filter games by player count
            games_mask = self.df.apply(
                lambda row: sum(1 for pos in range(1, 5)
                              if f'p{pos}_strategy' in self.df.columns
                              and not pd.isna(row[f'p{pos}_strategy'])) == player_count,
                axis=1
            )
            games_subset = self.df[games_mask]

            if len(games_subset) == 0:
                note = Paragraph(f"<i>No {player_count}-player games found in dataset.</i>", self.styles['BodyText'])
                self.story.append(note)
                self.story.append(Spacer(1, 0.2*inch))
                continue

            # Track matchups: strategy -> opponent_combination -> [wins, games]
            matchups = {}

            for idx, row in games_subset.iterrows():
                # Get all strategies in this game
                game_strategies = []
                for pos in range(1, player_count + 1):
                    strategy = row[f'p{pos}_strategy']
                    if not pd.isna(strategy):
                        game_strategies.append((pos, strategy))

                winner_pos = row.get('winner_position')
                if pd.isna(winner_pos):
                    continue

                # For each strategy, record its opponents and whether it won
                for pos, strategy in game_strategies:
                    if strategy not in matchups:
                        matchups[strategy] = {}

                    # Get opponent strategies (sorted for consistency)
                    opponents = tuple(sorted([s for p, s in game_strategies if p != pos]))

                    if opponents not in matchups[strategy]:
                        matchups[strategy][opponents] = {'wins': 0, 'games': 0}

                    matchups[strategy][opponents]['games'] += 1
                    if pos == winner_pos:
                        matchups[strategy][opponents]['wins'] += 1

            # Create table for this player count
            if matchups:
                # Sort strategies and get top matchups
                table_data = [['Strategy', 'vs Opponents', 'Games', 'Wins', 'Win Rate']]

                for strategy in sorted(matchups.keys()):
                    # Get all opponent combinations for this strategy
                    opponent_records = []
                    for opponents, record in matchups[strategy].items():
                        wins = record['wins']
                        games = record['games']
                        win_rate = (wins / games * 100) if games > 0 else 0
                        opponent_records.append((opponents, games, wins, win_rate))

                    # Sort by number of games (most common matchups first)
                    opponent_records.sort(key=lambda x: x[1], reverse=True)

                    # Add top 3 matchups for this strategy
                    for i, (opponents, games, wins, win_rate) in enumerate(opponent_records[:3]):
                        opponent_str = ' + '.join(opponents)
                        if len(opponent_str) > 40:
                            opponent_str = opponent_str[:37] + '...'

                        table_data.append([
                            strategy if i == 0 else '',  # Only show strategy name once
                            opponent_str,
                            f"{games}",
                            f"{wins}",
                            f"{win_rate:.1f}%"
                        ])

                table = Table(table_data, colWidths=[1.8*inch, 2.2*inch, 0.8*inch, 0.7*inch, 1*inch])
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('ALIGN', (0, 1), (0, -1), 'LEFT'),
                    ('ALIGN', (1, 1), (1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
                ]))

                self.story.append(table)
                self.story.append(Spacer(1, 0.3*inch))

            # Add summary stats for this player count
            total_games = len(games_subset)
            avg_score = games_subset['winner_score'].mean() if 'winner_score' in games_subset.columns else 0

            summary = Paragraph(
                f"<i>Total {player_count}P games: {total_games:,} | Avg winning score: {avg_score:.1f} VP</i>",
                self.styles['BodyText']
            )
            self.story.append(summary)
            self.story.append(Spacer(1, 0.3*inch))

    def _add_strategy_analysis(self):
        """Add strategy performance analysis with visualization."""
        self.story.append(Paragraph("Strategy Performance Analysis", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate win rates
        strategy_wins = self.df['winner_strategy'].value_counts().sort_index()
        total_games = len(self.df)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Bar chart
        strategies = strategy_wins.index
        win_rates = (strategy_wins.values / total_games * 100)
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(strategies)))

        ax1.bar(range(len(strategies)), win_rates, color=colors_list)
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate by Strategy')
        ax1.set_xticks(range(len(strategies)))
        ax1.set_xticklabels(strategies, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Pie chart
        ax2.pie(win_rates, labels=strategies, autopct='%1.1f%%', colors=colors_list, startangle=90)
        ax2.set_title('Win Distribution')

        plt.tight_layout()
        chart_path = Path(self.temp_dir) / 'strategy_performance.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(str(chart_path), width=6.5*inch, height=2.7*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))

        # Table of statistics
        data = [['Strategy', 'Wins', 'Win Rate', 'Avg VP']]

        # Calculate average VP per strategy
        strategy_vp = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            score_col = f'p{pos}_final_score'
            for strategy in strategies:
                mask = self.df[strategy_col] == strategy
                scores = self.df.loc[mask, score_col]
                if strategy not in strategy_vp:
                    strategy_vp[strategy] = []
                strategy_vp[strategy].extend(scores.tolist())

        for strategy, wins in strategy_wins.items():
            win_rate = (wins / total_games) * 100
            avg_vp = np.mean(strategy_vp[strategy]) if strategy in strategy_vp else 0
            data.append([strategy, f"{wins:,}", f"{win_rate:.1f}%", f"{avg_vp:.1f}"])

        table = Table(data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(PageBreak())

    def _add_position_analysis(self):
        """Add position-based performance analysis."""
        self.story.append(Paragraph("Position-Based Performance", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes how each strategy performs from different starting positions (1-4).",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate position win rates for each strategy
        strategies = sorted(self.df['winner_strategy'].unique())
        positions = [1, 2, 3, 4]

        position_data = {strategy: [0, 0, 0, 0] for strategy in strategies}

        for pos in positions:
            strategy_col = f'p{pos}_strategy'
            for strategy in strategies:
                mask = (self.df[strategy_col] == strategy) & (self.df['winner_position'] == pos)
                wins = mask.sum()
                total = (self.df[strategy_col] == strategy).sum()
                win_rate = (wins / total * 100) if total > 0 else 0
                position_data[strategy][pos - 1] = win_rate

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([position_data[s] for s in strategies])

        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

        # Set ticks
        ax.set_xticks(np.arange(4))
        ax.set_yticks(np.arange(len(strategies)))
        ax.set_xticklabels(['Position 1', 'Position 2', 'Position 3', 'Position 4'])
        ax.set_yticklabels(strategies)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Win Rate (%)', rotation=270, labelpad=20)

        # Add text annotations
        for i in range(len(strategies)):
            for j in range(4):
                text = ax.text(j, i, f'{data_matrix[i, j]:.1f}%',
                             ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Strategy Win Rate by Starting Position')
        plt.tight_layout()

        chart_path = Path(self.temp_dir) / 'position_heatmap.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(str(chart_path), width=6.5*inch, height=3.5*inch)
        self.story.append(img)
        self.story.append(PageBreak())

    def _add_vp_analysis(self):
        """Add VP distribution analysis."""
        self.story.append(Paragraph("Victory Point Analysis", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        # Gather all scores by strategy
        strategy_scores = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            score_col = f'p{pos}_final_score'
            # Skip if columns don't exist
            if strategy_col not in self.df.columns or score_col not in self.df.columns:
                continue
            for _, row in self.df.iterrows():
                strategy = row[strategy_col]
                score = row[score_col]
                # Skip NaN values
                if pd.isna(strategy) or pd.isna(score):
                    continue
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(score)

        # Create box plot
        fig, ax = plt.subplots(figsize=(12, 6))

        strategies = sorted(strategy_scores.keys())
        data_to_plot = [strategy_scores[s] for s in strategies]

        bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True)

        # Color boxes
        colors_list = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)

        ax.set_xlabel('Strategy')
        ax.set_ylabel('Victory Points')
        ax.set_title('VP Distribution by Strategy')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        chart_path = Path(self.temp_dir) / 'vp_distribution.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(str(chart_path), width=6.5*inch, height=3.5*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))

        # Statistics table
        data = [['Strategy', 'Mean VP', 'Std Dev', 'Min', 'Max', 'Median']]
        for strategy in strategies:
            scores = strategy_scores[strategy]
            data.append([
                strategy,
                f"{np.mean(scores):.1f}",
                f"{np.std(scores):.1f}",
                f"{np.min(scores):.0f}",
                f"{np.max(scores):.0f}",
                f"{np.median(scores):.1f}"
            ])

        table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 0.8*inch, 0.8*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        self.story.append(table)
        self.story.append(PageBreak())

    def _add_score_difference_analysis(self):
        """Add score difference analysis."""
        self.story.append(Paragraph("Score Differential Analysis", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Score differential measures the gap between the winner and other players, "
            "indicating game competitiveness and strategy dominance.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.hist(self.df['score_difference'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.df['score_difference'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.df["score_difference"].mean():.1f}')
        ax.axvline(self.df['score_difference'].median(), color='green', linestyle='--',
                   label=f'Median: {self.df["score_difference"].median():.1f}')

        ax.set_xlabel('Score Difference (Winner - Last Place)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Score Differences')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        chart_path = Path(self.temp_dir) / 'score_diff.png'
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(str(chart_path), width=6*inch, height=2.5*inch)
        self.story.append(img)
        self.story.append(Spacer(1, 0.2*inch))

        # Statistics
        stats_text = f"""
        <b>Mean Difference:</b> {self.df['score_difference'].mean():.2f} VP<br/>
        <b>Median Difference:</b> {self.df['score_difference'].median():.2f} VP<br/>
        <b>Std Deviation:</b> {self.df['score_difference'].std():.2f} VP<br/>
        <b>Min Difference:</b> {self.df['score_difference'].min():.0f} VP<br/>
        <b>Max Difference:</b> {self.df['score_difference'].max():.0f} VP<br/>
        """
        self.story.append(Paragraph(stats_text, self.styles['BodyText']))
        self.story.append(PageBreak())

    def _add_key_insights(self):
        """Add key insights and recommendations."""
        self.story.append(Paragraph("Key Insights & Recommendations", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate insights
        winner_stats = self.df['winner_strategy'].value_counts()
        best_strategy = winner_stats.index[0]
        best_rate = (winner_stats.iloc[0] / len(self.df)) * 100

        worst_strategy = winner_stats.index[-1]
        worst_rate = (winner_stats.iloc[-1] / len(self.df)) * 100

        # Strategy VP averages
        strategy_vp = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            score_col = f'p{pos}_final_score'
            for strategy in winner_stats.index:
                mask = self.df[strategy_col] == strategy
                scores = self.df.loc[mask, score_col]
                if strategy not in strategy_vp:
                    strategy_vp[strategy] = []
                strategy_vp[strategy].extend(scores.tolist())

        highest_vp_strategy = max(strategy_vp.items(), key=lambda x: np.mean(x[1]))[0]
        highest_vp = np.mean(strategy_vp[highest_vp_strategy])

        insights = [
            f"<b>1. Strategy Effectiveness:</b> {best_strategy} demonstrated the highest win rate at "
            f"{best_rate:.1f}%, significantly outperforming other strategies. This suggests that "
            f"its approach to the game mechanics is most effective.",

            f"<b>2. Consistency Matters:</b> {highest_vp_strategy} achieved the highest average VP "
            f"({highest_vp:.1f}), indicating consistent performance even when not winning. "
            f"Consistency in scoring is as important as win optimization.",

            f"<b>3. Random Baseline:</b> The Random strategy serves as a baseline. Any strategy "
            f"performing below or near Random suggests suboptimal decision-making that could be improved.",

            f"<b>4. Positional Advantages:</b> Analysis of position-based win rates reveals whether "
            f"certain starting positions provide inherent advantages, which is important for game balance.",

            f"<b>5. Score Competitiveness:</b> The average score difference of "
            f"{self.df['score_difference'].mean():.1f} VP indicates the typical margin of victory, "
            f"suggesting game outcomes are {('decisive' if self.df['score_difference'].mean() > 15 else 'competitive')}.",

            f"<b>6. Strategy Diversity:</b> With {len(winner_stats)} strategies tested, the distribution "
            f"of wins shows the game's complexity and the viability of different approaches.",

            f"<b>Recommendations:</b><br/>"
            f"• Focus on implementing elements from {best_strategy} in human gameplay<br/>"
            f"• Consider game balance adjustments if positional advantages are too strong<br/>"
            f"• Study why {worst_strategy} underperforms ({worst_rate:.1f}% win rate)<br/>"
            f"• Develop hybrid strategies combining successful elements from top performers<br/>"
            f"• Use this data to inform rule adjustments or variant designs"
        ]

        for insight in insights:
            para = Paragraph(insight, self.styles['BodyText'])
            self.story.append(para)
            self.story.append(Spacer(1, 0.15*inch))

        # Final note
        self.story.append(Spacer(1, 0.3*inch))
        final = Paragraph(
            "<i>This analysis is based on AI vs AI simulations and provides statistical insights "
            "into strategy effectiveness. Human gameplay may yield different results due to "
            "psychological factors and adaptive play.</i>",
            self.styles['BodyText']
        )
        self.story.append(final)

    def _add_strategy_usage(self):
        """Add strategy usage frequency section."""
        self.story.append(PageBreak())

        title = Paragraph("Strategy Usage Frequency", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section shows how frequently each strategy was used across all games and positions.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate strategy usage
        strategy_usage = {}
        strategy_position_usage = {pos: {} for pos in range(1, 5)}

        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            if strategy_col not in self.df.columns:
                continue
            for strategy in self.df[strategy_col]:
                # Skip NaN values (from games with fewer players)
                if pd.isna(strategy):
                    continue
                if strategy not in strategy_usage:
                    strategy_usage[strategy] = 0
                strategy_usage[strategy] += 1

                if strategy not in strategy_position_usage[pos]:
                    strategy_position_usage[pos][strategy] = 0
                strategy_position_usage[pos][strategy] += 1

        total_plays = sum(strategy_usage.values())

        # Create table
        table_data = [['Strategy', 'Total Plays', '% of Games', 'Pos 1', 'Pos 2', 'Pos 3', 'Pos 4']]

        for strategy in sorted(strategy_usage.keys()):
            count = strategy_usage[strategy]
            pct = count / total_plays * 100
            pos_counts = [str(strategy_position_usage[pos].get(strategy, 0)) for pos in range(1, 5)]
            table_data.append([
                strategy,
                f"{count:,}",
                f"{pct:.1f}%",
                pos_counts[0],
                pos_counts[1],
                pos_counts[2],
                pos_counts[3]
            ])

        table = Table(table_data, colWidths=[2*inch, 0.9*inch, 0.9*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ]))

        self.story.append(table)
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: In balanced simulations, each strategy appears equally across all positions.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)

    def _add_head_to_head_analysis(self):
        """Add head-to-head win rate matrix."""
        self.story.append(PageBreak())

        title = Paragraph("Head-to-Head Win Rates", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This matrix shows win rates when Strategy A (row) plays against Strategy B (column) "
            "in any position. Values represent the percentage of games won by the row strategy.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Build head-to-head matrix
        strategy_usage = {}
        for pos in range(1, 5):
            col = f'p{pos}_strategy'
            if col not in self.df.columns:
                continue
            for strategy in self.df[col]:
                # Skip NaN values
                if pd.isna(strategy):
                    continue
                strategy_usage[strategy] = True

        all_strat_names = sorted(strategy_usage.keys())
        h2h_matrix = {s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names} for s1 in all_strat_names}

        # Calculate head-to-head stats
        for idx, row in self.df.iterrows():
            winner_strategy = row['winner_strategy']
            winner_pos = row['winner_position']

            # Skip if winner strategy is NaN
            if pd.isna(winner_strategy):
                continue

            # Collect strategies, skipping NaN values
            strategies = []
            for pos in range(1, 5):
                col = f'p{pos}_strategy'
                if col in self.df.columns and not pd.isna(row[col]):
                    strategies.append((pos, row[col]))

            for pos, opponent in strategies:
                if pos != winner_pos:
                    h2h_matrix[winner_strategy][opponent]['wins'] += 1
                    h2h_matrix[winner_strategy][opponent]['games'] += 1
                    h2h_matrix[opponent][winner_strategy]['games'] += 1

        # Create heatmap
        matrix_data = []
        for s1 in all_strat_names:
            row_data = []
            for s2 in all_strat_names:
                if s1 == s2:
                    row_data.append(None)
                else:
                    games = h2h_matrix[s1][s2]['games']
                    if games > 0:
                        wins = h2h_matrix[s1][s2]['wins']
                        win_rate = wins / games * 100
                        row_data.append(win_rate)
                    else:
                        row_data.append(0)
            matrix_data.append(row_data)

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow([[v if v is not None else np.nan for v in row] for row in matrix_data],
                       cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(np.arange(len(all_strat_names)))
        ax.set_yticks(np.arange(len(all_strat_names)))
        ax.set_xticklabels([s[:10] for s in all_strat_names], rotation=45, ha='right')
        ax.set_yticklabels([s[:15] for s in all_strat_names])

        # Add text annotations
        for i in range(len(all_strat_names)):
            for j in range(len(all_strat_names)):
                if matrix_data[i][j] is not None:
                    text = ax.text(j, i, f"{matrix_data[i][j]:.0f}%",
                                 ha="center", va="center", color="black", fontsize=7)

        ax.set_title("Head-to-Head Win Rate Matrix (%)")
        fig.colorbar(im, ax=ax, label='Win Rate %')
        plt.tight_layout()

        img_path = f"{self.temp_dir}/h2h_matrix.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(img_path, width=6.5*inch, height=5.2*inch)
        self.story.append(img)

    def _add_matchup_analysis(self):
        """Add comprehensive matchup performance matrix."""
        self.story.append(PageBreak())

        title = Paragraph("Comprehensive Matchup Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "These matrices show how each strategy performs when facing different opponents. "
            "Each cell represents the win rate when the row strategy faces the column strategy as an opponent "
            "(averaged across all games where both strategies were present).",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Build comprehensive matchup data: strategy vs each opponent
        # For each strategy, track win rate when each other strategy is in the game as opponent
        strategy_usage = {}
        for pos in range(1, 5):
            col = f'p{pos}_strategy'
            if col not in self.df.columns:
                continue
            for strategy in self.df[col]:
                # Skip NaN values
                if pd.isna(strategy):
                    continue
                strategy_usage[strategy] = True

        all_strat_names = sorted(strategy_usage.keys())

        # Matrix: [my_strategy][opponent_strategy] = {'wins': X, 'games': Y}
        matchup_matrix = {s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names} for s1 in all_strat_names}

        for idx, row in self.df.iterrows():
            winner_pos = row['winner_position']
            winner_strategy = row['winner_strategy']

            # Skip if winner strategy is NaN
            if pd.isna(winner_strategy):
                continue

            # Collect strategies, skipping NaN values
            strategies = []
            for pos in range(1, 5):
                col = f'p{pos}_strategy'
                if col in self.df.columns and not pd.isna(row[col]):
                    strategies.append((pos, row[col]))

            # Skip games with missing data
            if len(strategies) == 0:
                continue

            # Build position to strategy mapping
            pos_to_strat = {pos: strat for pos, strat in strategies}

            # For each position, record performance against opponents
            for pos, my_strategy in strategies:
                did_win = (pos == winner_pos)

                # Against each opponent in this game
                for opp_pos, opponent in strategies:
                    if opp_pos != pos:
                        matchup_matrix[my_strategy][opponent]['games'] += 1
                        if did_win:
                            matchup_matrix[my_strategy][opponent]['wins'] += 1

        # Create heatmap for each strategy (show all 7 strategies in 2 pages)
        strategies_per_page = 4
        for page_num, start_idx in enumerate(range(0, len(all_strat_names), strategies_per_page)):
            if page_num > 0:
                self.story.append(PageBreak())

            end_idx = min(start_idx + strategies_per_page, len(all_strat_names))
            page_strategies = all_strat_names[start_idx:end_idx]

            # Create subplots for this page
            fig, axes = plt.subplots(2, 2, figsize=(11, 10))
            axes = axes.flatten()

            for i, strategy in enumerate(page_strategies):
                if i >= len(axes):
                    break

                ax = axes[i]

                # Calculate win rates against each opponent
                win_rates = []
                for opponent in all_strat_names:
                    games = matchup_matrix[strategy][opponent]['games']
                    if games > 0 and strategy != opponent:
                        wins = matchup_matrix[strategy][opponent]['wins']
                        win_rate = (wins / games) * 100
                    else:
                        win_rate = np.nan
                    win_rates.append(win_rate)

                # Create bar chart
                valid_indices = [j for j, wr in enumerate(win_rates) if not np.isnan(wr)]
                valid_names = [all_strat_names[j][:12] for j in valid_indices]
                valid_rates = [win_rates[j] for j in valid_indices]

                bars = ax.barh(range(len(valid_rates)), valid_rates, color='steelblue')
                ax.set_yticks(range(len(valid_rates)))
                ax.set_yticklabels(valid_names, fontsize=8)
                ax.set_xlabel('Win Rate (%)', fontsize=9)
                ax.set_title(f'{strategy[:20]}', fontsize=10, fontweight='bold')
                ax.set_xlim(0, 100)
                ax.grid(axis='x', alpha=0.3)
                ax.axvline(x=25, color='red', linestyle='--', alpha=0.5, linewidth=1)

                # Add value labels
                for bar, rate in zip(bars, valid_rates):
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                           f'{rate:.1f}%', ha='left', va='center', fontsize=7)

            # Hide unused subplots
            for i in range(len(page_strategies), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            img_path = f"{self.temp_dir}/matchup_analysis_page{page_num + 1}.png"
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            if page_num == 0:
                subtitle = Paragraph("<b>Win Rate vs Each Opponent Strategy (Part 1)</b>", self.styles['BodyText'])
                self.story.append(subtitle)
                self.story.append(Spacer(1, 0.1*inch))

            img = Image(img_path, width=7*inch, height=6.5*inch)
            self.story.append(img)

            if page_num == 0 and len(all_strat_names) > strategies_per_page:
                self.story.append(Spacer(1, 0.2*inch))

        # Add summary heatmap showing all strategies at once
        self.story.append(PageBreak())
        subtitle = Paragraph("<b>Complete Matchup Heatmap</b>", self.styles['CustomHeading'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        summary_text = Paragraph(
            "This heatmap shows the complete matchup matrix. Each cell shows the win rate of the row "
            "strategy when facing the column strategy as an opponent. Darker green = higher win rate.",
            self.styles['BodyText']
        )
        self.story.append(summary_text)
        self.story.append(Spacer(1, 0.3*inch))

        # Create full heatmap
        matrix_data = []
        for s1 in all_strat_names:
            row_data = []
            for s2 in all_strat_names:
                if s1 == s2:
                    row_data.append(np.nan)
                else:
                    games = matchup_matrix[s1][s2]['games']
                    if games > 0:
                        wins = matchup_matrix[s1][s2]['wins']
                        win_rate = (wins / games) * 100
                        row_data.append(win_rate)
                    else:
                        row_data.append(np.nan)
            matrix_data.append(row_data)

        fig, ax = plt.subplots(figsize=(10, 9))

        # Create masked array for NaN values
        masked_data = np.ma.array(matrix_data, mask=np.isnan(matrix_data))

        im = ax.imshow(masked_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

        # Labels
        ax.set_xticks(np.arange(len(all_strat_names)))
        ax.set_yticks(np.arange(len(all_strat_names)))
        ax.set_xticklabels([s[:12] for s in all_strat_names], rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([s[:15] for s in all_strat_names], fontsize=9)

        # Add text annotations
        for i in range(len(all_strat_names)):
            for j in range(len(all_strat_names)):
                if not np.isnan(matrix_data[i][j]):
                    text = ax.text(j, i, f"{matrix_data[i][j]:.0f}",
                                 ha="center", va="center", color="black", fontsize=7)

        ax.set_title("Strategy vs Opponent Win Rate Matrix (%)", fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel("Opponent Strategy", fontsize=10, fontweight='bold')
        ax.set_ylabel("My Strategy", fontsize=10, fontweight='bold')

        cbar = fig.colorbar(im, ax=ax, label='Win Rate %')
        cbar.ax.tick_params(labelsize=9)

        plt.tight_layout()

        img_path = f"{self.temp_dir}/matchup_heatmap_full.png"
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()

        img = Image(img_path, width=7*inch, height=6.2*inch)
        self.story.append(img)

    def _add_complete_combination_matrices(self):
        """Add complete matchup matrices showing all strategies vs all 343 opponent combinations."""
        self.story.append(PageBreak())

        title = Paragraph("Complete Opponent Combination Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section shows each strategy's performance against ALL possible 3-opponent combinations. "
            "First, individual bar charts show each strategy's performance. Then, a comprehensive matrix "
            "shows all strategies side-by-side for easy comparison. Combinations with Random opponents are shown first.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Get all strategies
        strategy_usage = {}
        for pos in range(1, 5):
            col = f'p{pos}_strategy'
            if col not in self.df.columns:
                continue
            for strategy in self.df[col]:
                # Skip NaN values
                if pd.isna(strategy):
                    continue
                strategy_usage[strategy] = True
        all_strat_names = sorted(strategy_usage.keys())

        # Build matchup data for ALL strategies against ALL opponent combinations
        # Structure: {strategy: {opponent_tuple: {'wins': X, 'games': Y}}}
        all_matchup_data = {s: {} for s in all_strat_names}

        for idx, row in self.df.iterrows():
            winner_pos = row['winner_position']

            # Collect strategies, skipping NaN values
            strategies = []
            for pos in range(1, 5):
                col = f'p{pos}_strategy'
                if col in self.df.columns:
                    strat = row[col]
                    if not pd.isna(strat):
                        strategies.append((pos, strat))

            # Skip games with fewer players (we need exactly 4 for combination analysis)
            if len(strategies) != 4:
                continue

            # Convert to position-indexed list for easier access
            strat_by_pos = {pos: strat for pos, strat in strategies}

            # For each position
            for pos, my_strategy in strategies:
                # Get the opponents (sorted tuple for consistency)
                opponents = tuple(sorted([strat for p, strat in strategies if p != pos]))

                if opponents not in all_matchup_data[my_strategy]:
                    all_matchup_data[my_strategy][opponents] = {'wins': 0, 'games': 0}

                all_matchup_data[my_strategy][opponents]['games'] += 1
                if pos == winner_pos:
                    all_matchup_data[my_strategy][opponents]['wins'] += 1

        # Get all unique opponent combinations across all strategies
        all_combinations = set()
        for strategy_data in all_matchup_data.values():
            all_combinations.update(strategy_data.keys())

        # Sort combinations: Random combinations first, then alphabetically
        def sort_key(combo):
            has_random = 'Random' in combo
            return (not has_random, combo)  # False sorts before True, so Random combos come first

        all_combinations = sorted(list(all_combinations), key=sort_key)

        if not all_combinations:
            note = Paragraph("No matchup data available", self.styles['BodyText'])
            self.story.append(note)
            return

        # ===== PART 1: Individual bar charts for each strategy =====
        subtitle1 = Paragraph("<b>Individual Strategy Performance Charts</b>", self.styles['CustomHeading'])
        self.story.append(subtitle1)
        self.story.append(Spacer(1, 0.2*inch))

        for strategy_idx, my_strategy in enumerate(all_strat_names):
            self.story.append(PageBreak())

            # Strategy title
            strat_title = Paragraph(
                f"<b>{my_strategy}</b> - Performance vs All Opponent Combinations",
                self.styles['CustomHeading']
            )
            self.story.append(strat_title)
            self.story.append(Spacer(1, 0.2*inch))

            # Get matchup data for this strategy
            matchup_list = []
            for opponents in all_combinations:
                if opponents in all_matchup_data[my_strategy]:
                    stats = all_matchup_data[my_strategy][opponents]
                    if stats['games'] > 0:
                        win_rate = (stats['wins'] / stats['games']) * 100
                        matchup_list.append({
                            'opponents': opponents,
                            'wins': stats['wins'],
                            'games': stats['games'],
                            'win_rate': win_rate
                        })

            if not matchup_list:
                note = Paragraph(f"No matchup data available for {my_strategy}", self.styles['BodyText'])
                self.story.append(note)
                continue

            # Sort by win rate (descending)
            matchup_list.sort(key=lambda x: x['win_rate'], reverse=True)

            num_matchups = len(matchup_list)

            # Create figure with appropriate size
            fig_width = 16
            fig_height = max(12, num_matchups * 0.08)

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Prepare data for bar chart
            win_rates = [m['win_rate'] for m in matchup_list]
            games_counts = [m['games'] for m in matchup_list]
            opponent_labels = [
                f"{m['opponents'][0][:8]}, {m['opponents'][1][:8]}, {m['opponents'][2][:8]}"
                for m in matchup_list
            ]

            # Create horizontal bar chart with color coding
            colors = plt.cm.RdYlGn(np.array(win_rates) / 100)

            y_pos = np.arange(len(win_rates))
            bars = ax.barh(y_pos, win_rates, color=colors, edgecolor='black', linewidth=0.5)

            # Labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(opponent_labels, fontsize=6)
            ax.set_xlabel('Win Rate (%)', fontsize=10, fontweight='bold')
            ax.set_title(f'{my_strategy} vs All Opponent Combinations\n({num_matchups} unique matchups)',
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlim(0, 100)
            ax.axvline(x=25, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Random baseline (25%)')
            ax.grid(axis='x', alpha=0.3)
            ax.legend(fontsize=8)

            # Add win rate and game count annotations
            for i, (bar, rate, games) in enumerate(zip(bars, win_rates, games_counts)):
                width = bar.get_width()
                ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                       f'{rate:.1f}% ({games}g)',
                       ha='left', va='center', fontsize=5)

            plt.tight_layout()

            # Save with high DPI
            safe_name = my_strategy.replace(' ', '_').replace('/', '_').replace('\\', '_')
            img_path = f"{self.temp_dir}/complete_matchup_{safe_name}.png"
            plt.savefig(img_path, dpi=200, bbox_inches='tight')
            plt.close()

            # Add to PDF
            max_img_width = 7.5 * inch
            max_img_height = 10 * inch

            aspect = fig_width / fig_height
            if aspect > (max_img_width / max_img_height):
                img_width = max_img_width
                img_height = img_width / aspect
            else:
                img_height = max_img_height
                img_width = img_height * aspect

            img = Image(img_path, width=img_width, height=img_height)
            self.story.append(img)

            # Add summary stats
            self.story.append(Spacer(1, 0.2*inch))

            best_matchup = matchup_list[0]
            worst_matchup = matchup_list[-1]
            avg_win_rate = np.mean(win_rates)

            summary = Paragraph(
                f"<b>Summary for {my_strategy}:</b><br/>"
                f"• Total unique opponent combinations faced: {num_matchups}<br/>"
                f"• Average win rate across all matchups: {avg_win_rate:.1f}%<br/>"
                f"• Best matchup: vs [{', '.join([o[:12] for o in best_matchup['opponents']])}] "
                f"at {best_matchup['win_rate']:.1f}% ({best_matchup['wins']}/{best_matchup['games']} games)<br/>"
                f"• Worst matchup: vs [{', '.join([o[:12] for o in worst_matchup['opponents']])}] "
                f"at {worst_matchup['win_rate']:.1f}% ({worst_matchup['wins']}/{worst_matchup['games']} games)",
                self.styles['BodyText']
            )
            self.story.append(summary)

        # ===== PART 2: Comprehensive matrix showing all strategies =====
        self.story.append(PageBreak())

        subtitle2 = Paragraph("<b>Comprehensive Strategy Comparison Matrix</b>", self.styles['CustomHeading'])
        self.story.append(subtitle2)
        self.story.append(Spacer(1, 0.2*inch))

        matrix_intro = Paragraph(
            "This matrix shows all strategies side-by-side for easy comparison. "
            "Each row represents one opponent combination, each column represents a strategy. "
            "Combinations with Random opponents are shown first.",
            self.styles['BodyText']
        )
        self.story.append(matrix_intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Build matrix: rows = opponent combinations, columns = strategies
        matrix_data = []
        combination_labels = []

        for opponents in all_combinations:
            row_data = []
            for strategy in all_strat_names:
                if opponents in all_matchup_data[strategy]:
                    stats = all_matchup_data[strategy][opponents]
                    if stats['games'] > 0:
                        win_rate = (stats['wins'] / stats['games']) * 100
                        row_data.append(win_rate)
                    else:
                        row_data.append(np.nan)
                else:
                    row_data.append(np.nan)

            matrix_data.append(row_data)
            # Create short label for opponent combination
            label = f"{opponents[0][:4]},{opponents[1][:4]},{opponents[2][:4]}"
            combination_labels.append(label)

        matrix_data = np.array(matrix_data)

        # Create matrix visualization - split into multiple pages if needed
        rows_per_page = 100  # Show 100 combinations per page
        num_pages = int(np.ceil(len(all_combinations) / rows_per_page))

        for page_idx in range(num_pages):
            start_row = page_idx * rows_per_page
            end_row = min((page_idx + 1) * rows_per_page, len(all_combinations))

            page_matrix = matrix_data[start_row:end_row, :]
            page_labels = combination_labels[start_row:end_row]

            # Create figure
            fig_height = max(10, len(page_labels) * 0.12)
            fig, ax = plt.subplots(figsize=(12, fig_height))

            # Create heatmap
            im = ax.imshow(page_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

            # Set ticks and labels
            ax.set_xticks(np.arange(len(all_strat_names)))
            ax.set_yticks(np.arange(len(page_labels)))
            ax.set_xticklabels([s[:15] for s in all_strat_names], rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(page_labels, fontsize=5)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Win Rate (%)', rotation=270, labelpad=15, fontsize=10)
            cbar.ax.tick_params(labelsize=8)

            # Add text annotations with win percentages
            for i in range(len(page_labels)):
                for j in range(len(all_strat_names)):
                    value = page_matrix[i, j]
                    if not np.isnan(value):
                        # Choose text color based on background
                        text_color = 'white' if value < 30 or value > 70 else 'black'
                        text = ax.text(j, i, f'{value:.0f}%',
                                     ha="center", va="center", color=text_color,
                                     fontsize=5, weight='bold')

            ax.set_title(f'Strategy Performance vs All Opponent Combinations '
                        f'(Page {page_idx + 1}/{num_pages}: '
                        f'Combinations {start_row + 1}-{end_row})',
                        fontsize=11, fontweight='bold', pad=10)
            ax.set_xlabel('Strategy', fontsize=10, fontweight='bold')
            ax.set_ylabel('Opponent Combination', fontsize=10, fontweight='bold')

            plt.tight_layout()

            # Save figure
            img_path = f"{self.temp_dir}/complete_matchup_matrix_page{page_idx + 1}.png"
            plt.savefig(img_path, dpi=200, bbox_inches='tight')
            plt.close()

            # Add to PDF
            if page_idx > 0:
                self.story.append(PageBreak())

            max_img_width = 7.5 * inch
            max_img_height = 9.5 * inch

            # Calculate dimensions preserving aspect ratio
            aspect = 12 / fig_height
            if aspect > (max_img_width / max_img_height):
                img_width = max_img_width
                img_height = img_width / aspect
            else:
                img_height = max_img_height
                img_width = img_height * aspect

            img = Image(img_path, width=img_width, height=img_height)
            self.story.append(img)

        # Add summary statistics
        self.story.append(Spacer(1, 0.3*inch))

        # Calculate average win rates per strategy
        avg_win_rates = {}
        for j, strategy in enumerate(all_strat_names):
            strategy_rates = matrix_data[:, j]
            valid_rates = strategy_rates[~np.isnan(strategy_rates)]
            if len(valid_rates) > 0:
                avg_win_rates[strategy] = np.mean(valid_rates)
            else:
                avg_win_rates[strategy] = 0

        best_overall = max(avg_win_rates.items(), key=lambda x: x[1])
        worst_overall = min(avg_win_rates.items(), key=lambda x: x[1])

        summary = Paragraph(
            f"<b>Overall Summary:</b><br/>"
            f"• Total opponent combinations analyzed: {len(all_combinations)}<br/>"
            f"• Number of strategies: {len(all_strat_names)}<br/>"
            f"• Best overall strategy: {best_overall[0]} (avg {best_overall[1]:.1f}% across all matchups)<br/>"
            f"• Worst overall strategy: {worst_overall[0]} (avg {worst_overall[1]:.1f}% across all matchups)<br/>"
            f"<br/>"
            f"<i>Each cell shows the strategy's win rate when facing that specific combination of 3 opponents. "
            f"This provides the most detailed view of strategy performance across all possible game scenarios.</i>",
            self.styles['BodyText']
        )
        self.story.append(summary)


def generate_pdf_report(csv_file: str, output_file: str) -> str:
    """
    Generate PDF report from CSV file.

    Args:
        csv_file: Path to aggregated CSV file
        output_file: Path to save PDF report

    Returns:
        Path to generated PDF
    """
    print("\n" + "=" * 70)
    print("GENERATING PDF REPORT")
    print("=" * 70)
    print(f"\nLoading data from: {csv_file}")

    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df):,} games with {len(df.columns)} columns")

    print("Generating visualizations and report...")
    report = SimulationReport(df, output_file)
    report.generate()

    return output_file
