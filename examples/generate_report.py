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
        """Create custom paragraph styles and unified plot styling."""
        # Paragraph styles
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
            name='SubsectionTitle',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#2c5f2d'),
            spaceAfter=8,
            spaceBefore=8
        ))
        self.styles.add(ParagraphStyle(
            name='Insight',
            parent=self.styles['BodyText'],
            fontSize=11,
            textColor=colors.HexColor('#2c5f2d'),
            leftIndent=20,
            bulletIndent=10
        ))
        self.styles.add(ParagraphStyle(
            name='PlotExplanation',
            parent=self.styles['BodyText'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            leftIndent=10,
            spaceAfter=8,
            leading=14
        ))

        # Unified plot styling parameters
        self.plot_style = {
            'figsize_standard': (10, 5),
            'figsize_wide': (10, 6),
            'figsize_small': (8, 4),
            'dpi': 150,
            'title_fontsize': 12,
            'label_fontsize': 10,
            'tick_fontsize': 9,
            'legend_fontsize': 9,
        }

        # Unified color scheme
        self.colors = {
            'primary': '#1f4788',      # Dark blue
            'success': '#2c5f2d',      # Green
            'warning': '#8b4513',      # Brown
            'danger': '#8b0000',       # Dark red
            'highlight': '#ff8c00',    # Orange
            'secondary': '#4169e1',    # Royal blue
            'crafts': '#8b4513',
            'painting': '#1f4788',
            'sculpture': '#2c5f2d',
            'relic': '#8b0000',
            'nature': '#2c5f2d',
            'mythology': '#4169e1',
            'society': '#8b4513',
            'orientalism': '#ff8c00',
            'player2': '#1f4788',
            'player3': '#2c5f2d',
            'player4': '#8b4513',
        }

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

        # Build report sections in logical order
        self._add_title_page()
        self._add_table_of_contents()
        self._add_executive_summary()

        # Part 1: Overview
        self._add_game_overview_statistics()

        # Part 2: Strategy Performance
        self._add_strategy_performance_overview()

        # Part 3: Multi-Player & Matchups
        self._add_player_count_analysis()
        self._add_strategy_matchups_by_player_count()
        self._add_head_to_head_analysis()

        # Part 4: Gameplay Statistics
        self._add_works_and_vp_statistics()
        self._add_art_type_theme_analysis()

        # Part 5: Advanced Mode (if applicable)
        self._add_advanced_mode_statistics()

        # Part 6: Position & Insights
        self._add_position_analysis()
        self._add_key_insights()

        # Annexes: All Tabular Data
        self._add_annexes()

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

    def _add_table_of_contents(self):
        """Add comprehensive table of contents."""
        self.story.append(Paragraph("Table of Contents", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.3*inch))

        toc_items = [
            ("Executive Summary", "Overview of key findings and top performing strategies"),
            ("", ""),
            ("PART 1: Game Overview Statistics", ""),
            ("  • Winning Score Distribution", "Analysis of winning score patterns across all games"),
            ("  • Games by Player Count", "Breakdown of 2P, 3P, and 4P game distribution"),
            ("", ""),
            ("PART 2: Strategy Performance", ""),
            ("  • Overall Win Rates by Strategy", "Comparative win rate analysis for all strategies"),
            ("  • Average Scores by Strategy", "Score performance metrics per strategy"),
            ("  • Strategy Performance Summary", "Comprehensive statistics table"),
            ("  • Strategy Performance Analysis", "Visual breakdown with charts"),
            ("", ""),
            ("PART 3: Multi-Player & Matchups", ""),
            ("  • Multi-Player Game Analysis", "Performance broken down by 2P, 3P, and 4P games"),
            ("  • Strategy Performance by Player Count", "Win rate analysis for each player count"),
            ("  • Strategy Matchups by Player Count", "Head-to-head performance against opponent combinations"),
            ("  • Head-to-Head Analysis", "Direct strategy vs strategy comparisons"),
            ("", ""),
            ("PART 4: Gameplay Statistics", ""),
            ("  • Works & VP Expenditure Analysis", "Card usage and victory point economics"),
            ("  • VP Earned vs VP Spent", "Visual comparison by strategy"),
            ("  • Works Played vs Cards Discarded", "Card management efficiency"),
            ("  • Art Type & Theme Analysis", "Distribution of art types and themes by strategy"),
            ("", ""),
            ("PART 5: Advanced Mode Statistics", ""),
            ("  • Room Tile Acquisition", "Strategic tile acquisition patterns"),
            ("  • Advantage Card Selection", "Card preference analysis with heatmap"),
            ("  • Artist and Theme Analysis", "Works placement statistics"),
            ("", ""),
            ("PART 6: Position & Insights", ""),
            ("  • Position Analysis", "Starting position impact on win rates"),
            ("  • Key Insights", "Strategic recommendations and observations"),
        ]

        toc_data = []
        for section, description in toc_items:
            if section == "":
                toc_data.append(["", ""])
            else:
                toc_data.append([section, description])

        toc_table = Table(toc_data, colWidths=[2.8*inch, 3.7*inch])
        toc_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1f4788')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#666666')),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))

        self.story.append(toc_table)
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

    def _add_game_overview_statistics(self):
        """Add comprehensive game overview with visualizations."""
        self.story.append(PageBreak())

        title = Paragraph("PART 1: Game Overview Statistics", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section provides a comprehensive overview of all simulated games, analyzing score distributions "
            "and game configurations across different player counts.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Score distribution histogram
        subtitle = Paragraph("Winning Score Distribution", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This histogram displays the distribution of winning scores across all games. "
            "Each bar represents how many games were won with scores in that range. The red dashed line shows the "
            "mean (average) winning score, while the green dashed line shows the median (middle value).<br/><br/>"
            "<b>How to interpret:</b> A normal bell-curve distribution suggests balanced game mechanics. Outliers "
            "on either end may indicate dominant victories or close games. The proximity of mean and median indicates "
            "symmetry in score distribution.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        fig, ax = plt.subplots(figsize=self.plot_style['figsize_small'])
        self.df['winner_score'].hist(bins=30, ax=ax, color=self.colors['primary'], edgecolor='black', alpha=0.7)
        ax.set_xlabel('Winning Score (VP)', fontsize=self.plot_style['label_fontsize'])
        ax.set_ylabel('Frequency', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('Distribution of Winning Scores', fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=self.plot_style['tick_fontsize'])

        # Add mean and median lines
        mean_score = self.df['winner_score'].mean()
        median_score = self.df['winner_score'].median()
        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
        ax.axvline(median_score, color=self.colors['success'], linestyle='--', linewidth=2, label=f'Median: {median_score:.1f}')
        ax.legend(fontsize=self.plot_style['legend_fontsize'])

        img_path = f"{self.temp_dir}/score_distribution.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6*inch, height=3*inch))
        self.story.append(Spacer(1, 0.3*inch))

        # Player count distribution
        subtitle = Paragraph("Games by Player Count", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> The left chart shows the absolute number of games played with 2, 3, or 4 players, "
            "while the right pie chart shows the proportional distribution. Each player count may have different strategic "
            "dynamics.<br/><br/>"
            "<b>How to interpret:</b> This helps you understand the dataset composition. If one player count dominates, "
            "overall statistics may be skewed toward that configuration. Balanced representation across player counts "
            "ensures more generalizable conclusions.<br/><br/>"
            "<b>Why it matters:</b> Different player counts change game dynamics significantly - 2P games are more "
            "head-to-head tactical, while 4P games involve more complex multi-opponent positioning.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        # Count games by player count
        player_counts = []
        for idx, row in self.df.iterrows():
            num_players = sum(1 for pos in range(1, 5)
                            if f'p{pos}_strategy' in self.df.columns
                            and not pd.isna(row[f'p{pos}_strategy']))
            player_counts.append(num_players)

        player_count_series = pd.Series(player_counts)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.plot_style['figsize_standard'])

        # Bar chart
        counts = player_count_series.value_counts().sort_index()
        colors_list = [self.colors['player2'], self.colors['player3'], self.colors['player4']]
        ax1.bar(counts.index, counts.values, color=colors_list[:len(counts)], edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Number of Players', fontsize=self.plot_style['label_fontsize'])
        ax1.set_ylabel('Number of Games', fontsize=self.plot_style['label_fontsize'])
        ax1.set_title('Games Distribution by Player Count', fontsize=self.plot_style['title_fontsize'])
        ax1.set_xticks([2, 3, 4])
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(labelsize=self.plot_style['tick_fontsize'])

        # Add value labels on bars
        for i, (idx, val) in enumerate(zip(counts.index, counts.values)):
            ax1.text(idx, val + max(counts.values) * 0.02, f'{val:,}', ha='center', fontsize=9)

        # Pie chart
        ax2.pie(counts.values, labels=[f'{int(k)}P' for k in counts.index],
                autopct='%1.1f%%', colors=colors_list[:len(counts)],
                startangle=90, textprops={'fontsize': self.plot_style['tick_fontsize']})
        ax2.set_title('Player Count Proportions', fontsize=self.plot_style['title_fontsize'])

        img_path = f"{self.temp_dir}/player_count_dist.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=3*inch))
        self.story.append(Spacer(1, 0.3*inch))

    def _add_strategy_performance_overview(self):
        """Add visual overview of strategy performance."""
        self.story.append(PageBreak())

        title = Paragraph("PART 2: Strategy Performance Overview", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes the performance of each AI strategy across all games, comparing win rates and "
            "average scores. These metrics help identify which strategies are most effective overall and in specific contexts.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Calculate strategy statistics
        strategy_stats = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            score_col = f'p{pos}_final_score'

            if strategy_col not in self.df.columns or score_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                score = row[score_col]

                if pd.isna(strategy) or pd.isna(score):
                    continue

                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'wins': 0, 'games': 0, 'total_score': 0}

                strategy_stats[strategy]['games'] += 1
                strategy_stats[strategy]['total_score'] += score

                if row['winner_strategy'] == strategy:
                    strategy_stats[strategy]['wins'] += 1

        # Win rates bar chart
        subtitle = Paragraph("Overall Win Rates by Strategy", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This horizontal bar chart compares the win rate (percentage of games won) for each "
            "strategy across all games. The red dashed line represents the expected win rate of 25% for balanced 4-player "
            "games. Bars are color-coded: green for high performers (>30%), blue for average, and brown for low performers (<20%).<br/><br/>"
            "<b>How to interpret:</b> Strategies significantly above the 25% line are overperforming, while those below "
            "are underperforming. In balanced games, all strategies should cluster near 25%. Large deviations suggest "
            "strategic advantages or weaknesses.<br/><br/>"
            "<b>Why it matters:</b> This is the primary metric for strategy effectiveness. However, remember that raw win "
            "rates don't account for player count variations - see the Multi-Player Analysis section for player-count-specific rates.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        strategies = sorted(strategy_stats.keys())
        win_rates = [(strategy_stats[s]['wins'] / strategy_stats[s]['games'] * 100)
                     for s in strategies]

        fig, ax = plt.subplots(figsize=self.plot_style['figsize_standard'])
        bars = ax.barh(strategies, win_rates, color=self.colors['primary'], edgecolor='black', alpha=0.8)

        # Color code bars by performance
        for i, bar in enumerate(bars):
            if win_rates[i] > 30:
                bar.set_color(self.colors['success'])  # Green for high performers
            elif win_rates[i] < 20:
                bar.set_color(self.colors['warning'])  # Brown for low performers

        ax.set_xlabel('Win Rate (%)', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('Strategy Win Rates (Expected: 25% in 4-player games)',
                    fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.axvline(25, color='red', linestyle='--', linewidth=2, label='Expected (25%)')
        ax.legend(fontsize=self.plot_style['legend_fontsize'])
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(labelsize=self.plot_style['tick_fontsize'])

        # Add value labels
        for i, (strategy, rate) in enumerate(zip(strategies, win_rates)):
            ax.text(rate + 0.5, i, f'{rate:.1f}%', va='center', fontsize=self.plot_style['tick_fontsize'])

        img_path = f"{self.temp_dir}/win_rates.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=4*inch))
        self.story.append(Spacer(1, 0.3*inch))

        # Average scores bar chart
        subtitle = Paragraph("Average Scores by Strategy", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This chart displays the average score achieved by each strategy across all games, "
            "regardless of whether they won or lost. This provides insight into overall point accumulation effectiveness.<br/><br/>"
            "<b>How to interpret:</b> Higher average scores indicate more consistent point generation. A strategy with high "
            "average scores but moderate win rate might be competitive but lose close games. Conversely, low average scores "
            "with high win rates suggest the strategy wins decisively when it does win.<br/><br/>"
            "<b>Why it matters:</b> Average score complements win rate analysis. Strategies that score consistently high "
            "are more reliable, while those with variable scores may be more matchup-dependent or risky.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        avg_scores = [(strategy_stats[s]['total_score'] / strategy_stats[s]['games'])
                      for s in strategies]

        fig, ax = plt.subplots(figsize=self.plot_style['figsize_standard'])
        bars = ax.barh(strategies, avg_scores, color=self.colors['success'], edgecolor='black', alpha=0.8)
        ax.set_xlabel('Average Score (VP)', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('Average Scores by Strategy', fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(labelsize=self.plot_style['tick_fontsize'])

        # Add value labels
        for i, (strategy, score) in enumerate(zip(strategies, avg_scores)):
            ax.text(score + 0.5, i, f'{score:.1f}', va='center', fontsize=self.plot_style['tick_fontsize'])

        img_path = f"{self.temp_dir}/avg_scores.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=4*inch))
        self.story.append(Spacer(1, 0.3*inch))

        note = Paragraph(
            "<i>Note: Detailed tabular statistics are available in the Annexes section at the end of this report.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)
        self.story.append(Spacer(1, 0.2*inch))

    def _add_art_type_theme_analysis(self):
        """Add comprehensive art type and theme analysis with visualizations."""
        self.story.append(PageBreak())

        title = Paragraph("Art Type & Theme Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes which art types (Crafts, Painting, Sculpture, Relic) and themes (Nature, Mythology, "
            "Society, Orientalism) are preferred by each strategy. These patterns reveal strategic focus areas and can "
            "indicate whether strategies successfully pursue room type/theme bonuses or fashion trends.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Collect data by strategy
        strategy_art_data = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            if strategy_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                if pd.isna(strategy):
                    continue

                if strategy not in strategy_art_data:
                    strategy_art_data[strategy] = {
                        'crafts': [], 'painting': [], 'sculpture': [], 'relic': [],
                        'nature': [], 'mythology': [], 'society': [], 'orientalism': []
                    }

                # Collect art types
                for art_type in ['crafts', 'painting', 'sculpture', 'relic']:
                    col = f'p{pos}_works_{art_type}'
                    if col in self.df.columns:
                        strategy_art_data[strategy][art_type].append(row[col])

                # Collect themes
                for theme in ['nature', 'mythology', 'society', 'orientalism']:
                    col = f'p{pos}_works_{theme}'
                    if col in self.df.columns:
                        strategy_art_data[strategy][theme].append(row[col])

        if strategy_art_data:
            # Art Type stacked bar chart
            subtitle = Paragraph("Works by Art Type (Stacked by Strategy)", self.styles['Heading2'])
            self.story.append(subtitle)
            self.story.append(Spacer(1, 0.1*inch))

            explanation = Paragraph(
                "<b>What this shows:</b> This stacked bar chart displays the composition of works played by each strategy, "
                "broken down by art type (Crafts, Painting, Sculpture, Relic). Each colored segment represents the average "
                "number of works of that type placed per game.<br/><br/>"
                "<b>How to interpret:</b> Total bar height shows overall productivity (total works placed), while segment "
                "proportions reveal art type preferences. Balanced distributions suggest flexible strategies, while skewed "
                "distributions indicate specialization in specific room types (since room types correspond to art types).<br/><br/>"
                "<b>Why it matters:</b> Art type distribution reveals whether strategies successfully pursue room-type bonuses. "
                "Strategies focused on specific types should show clear segment dominance.",
                self.styles['PlotExplanation']
            )
            self.story.append(explanation)
            self.story.append(Spacer(1, 0.15*inch))

            strategies = sorted(strategy_art_data.keys())
            art_types = ['crafts', 'painting', 'sculpture', 'relic']
            art_type_data = {art_type: [np.mean(strategy_art_data[s][art_type])
                                         if strategy_art_data[s][art_type] else 0
                                         for s in strategies]
                            for art_type in art_types}

            fig, ax = plt.subplots(figsize=self.plot_style['figsize_wide'])
            x = np.arange(len(strategies))
            width = 0.6
            bottom = np.zeros(len(strategies))

            for art_type in art_types:
                ax.bar(x, art_type_data[art_type], width, label=art_type.capitalize(),
                      bottom=bottom, color=self.colors[art_type], edgecolor='black', alpha=0.8)
                bottom += art_type_data[art_type]

            ax.set_ylabel('Average Works Played', fontsize=self.plot_style['label_fontsize'])
            ax.set_title('Art Type Distribution by Strategy', fontsize=self.plot_style['title_fontsize'], pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=self.plot_style['tick_fontsize'])
            ax.legend(loc='upper left', fontsize=self.plot_style['legend_fontsize'])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='y', labelsize=self.plot_style['tick_fontsize'])

            img_path = f"{self.temp_dir}/art_types_stacked.png"
            plt.tight_layout()
            plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
            plt.close()

            self.story.append(Image(img_path, width=6.5*inch, height=4.5*inch))
            self.story.append(Spacer(1, 0.3*inch))

            # Theme stacked bar chart
            subtitle = Paragraph("Works by Theme (Stacked by Strategy)", self.styles['Heading2'])
            self.story.append(subtitle)
            self.story.append(Spacer(1, 0.1*inch))

            explanation = Paragraph(
                "<b>What this shows:</b> This stacked bar chart displays theme distribution (Nature, Mythology, Society, "
                "Orientalism) for works placed by each strategy. Similar to art types, this reveals thematic preferences.<br/><br/>"
                "<b>How to interpret:</b> Theme distribution indicates whether strategies pursue room-theme bonuses or "
                "fashion trend objectives. Balanced themes suggest flexible play adapting to available cards, while skewed "
                "distributions suggest specialization in specific thematic focuses.<br/><br/>"
                "<b>Why it matters:</b> Theme bonuses can provide significant VP advantages. Strategies that successfully "
                "concentrate on specific themes should show clear dominance in those segments, correlating with higher scores "
                "when fashion trends align with their thematic focus.",
                self.styles['PlotExplanation']
            )
            self.story.append(explanation)
            self.story.append(Spacer(1, 0.15*inch))

            themes = ['nature', 'mythology', 'society', 'orientalism']
            theme_data = {theme: [np.mean(strategy_art_data[s][theme])
                                  if strategy_art_data[s][theme] else 0
                                  for s in strategies]
                         for theme in themes}

            fig, ax = plt.subplots(figsize=self.plot_style['figsize_wide'])
            bottom = np.zeros(len(strategies))

            for theme in themes:
                ax.bar(x, theme_data[theme], width, label=theme.capitalize(),
                      bottom=bottom, color=self.colors[theme], edgecolor='black', alpha=0.8)
                bottom += theme_data[theme]

            ax.set_ylabel('Average Works Played', fontsize=self.plot_style['label_fontsize'])
            ax.set_title('Theme Distribution by Strategy', fontsize=self.plot_style['title_fontsize'], pad=10)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=self.plot_style['tick_fontsize'])
            ax.legend(loc='upper left', fontsize=self.plot_style['legend_fontsize'])
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='y', labelsize=self.plot_style['tick_fontsize'])

            img_path = f"{self.temp_dir}/themes_stacked.png"
            plt.tight_layout()
            plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
            plt.close()

            self.story.append(Image(img_path, width=6.5*inch, height=4.5*inch))
            self.story.append(Spacer(1, 0.3*inch))

    def _add_player_count_analysis(self):
        """Add analysis broken down by player count."""
        self.story.append(PageBreak())

        title = Paragraph("PART 3: Multi-Player Game Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section provides detailed analysis broken down by player count (2P, 3P, and 4P games). "
            "Game dynamics change significantly with different player counts: 2-player games are more tactical and direct, "
            "3-player games introduce triangular dynamics, and 4-player games involve complex multi-opponent strategies. "
            "Understanding performance across these configurations is crucial for evaluating strategy robustness.",
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

        # Player count summary will be visualized in the overview section

        # Strategy performance by player count
        subtitle = Paragraph("Strategy Performance by Player Count", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This table displays win rates for each strategy broken down by player count (2P, 3P, 4P). "
            "Win rates are normalized to show performance relative to expectation: 100% means the strategy won exactly as "
            "often as expected (50% in 2P, 33% in 3P, 25% in 4P), values above 100% indicate overperformance.<br/><br/>"
            "<b>How to interpret:</b> Compare horizontally to see how a strategy adapts to different player counts. "
            "Some strategies excel in head-to-head 2P games but struggle in 4P games, or vice versa. A truly robust strategy "
            "performs well across all player counts (consistently above 100%).<br/><br/>"
            "<b>Why it matters:</b> This reveals strategic versatility. Strategies that only perform well at specific player "
            "counts may have exploitable patterns or require specific opponent configurations to succeed.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.2*inch))

        # Create win rate visualization
        all_strategies = set()
        for data in player_count_data.values():
            all_strategies.update(data['strategies'].keys())

        strategies = sorted(all_strategies)
        player_counts = [2, 3, 4]

        # Build data matrix for visualization
        win_rate_matrix = []
        for strategy in strategies:
            strategy_rates = []
            for pc in player_counts:
                if pc in player_count_data:
                    wins = player_count_data[pc]['strategies'].get(strategy, 0)
                    games = player_count_data[pc]['games']
                    expected_wins = games / pc
                    win_rate = (wins / expected_wins * 100) if expected_wins > 0 else 0
                    strategy_rates.append(win_rate)
                else:
                    strategy_rates.append(0)
            win_rate_matrix.append(strategy_rates)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=self.plot_style['figsize_wide'])
        x = np.arange(len(strategies))
        width = 0.25

        bars1 = ax.bar(x - width, [row[0] for row in win_rate_matrix], width,
                      label='2P Games', color=self.colors['player2'], edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x, [row[1] for row in win_rate_matrix], width,
                      label='3P Games', color=self.colors['player3'], edgecolor='black', alpha=0.8)
        bars3 = ax.bar(x + width, [row[2] for row in win_rate_matrix], width,
                      label='4P Games', color=self.colors['player4'], edgecolor='black', alpha=0.8)

        ax.set_ylabel('Normalized Win Rate (%)', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('Strategy Performance Across Player Counts', fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=self.plot_style['tick_fontsize'])
        ax.legend(fontsize=self.plot_style['legend_fontsize'])
        ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Expected (100%)')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='y', labelsize=self.plot_style['tick_fontsize'])

        img_path = f"{self.temp_dir}/player_count_performance.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=4.5*inch))
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: Win rates are normalized by expected performance (100% = performs as expected for that player count).</i>",
            self.styles['BodyText']
        )
        self.story.append(note)

    def _add_works_and_vp_statistics(self):
        """Add comprehensive works played/discarded and VP spending statistics."""
        self.story.append(PageBreak())

        title = Paragraph("PART 4: Works & VP Expenditure Analysis", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes the game economy: how strategies manage their card hands, how many works they successfully "
            "place versus discard, and how they balance VP earning versus spending. Understanding these patterns reveals "
            "strategic decision-making and resource management efficiency.",
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

        # VP Earned vs Spent Visualization
        subtitle = Paragraph("VP Earned vs VP Spent by Strategy", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This grouped bar chart compares average Victory Points (VP) earned versus VP spent for "
            "each strategy. Green bars show VP earned from placing works, while brown bars show VP spent on room tiles and "
            "other strategic purchases.<br/><br/>"
            "<b>How to interpret:</b> Ideally, VP earned should significantly exceed VP spent - this indicates efficient "
            "resource management. Strategies with high spending but low earnings may be over-investing in infrastructure. "
            "The gap between earned and spent represents net VP contribution to final score.<br/><br/>"
            "<b>Why it matters:</b> This reveals economic efficiency. A winning strategy must balance investment in "
            "infrastructure (room tiles) with actual point generation from works. Over-spending on tiles can leave you "
            "point-starved even with a perfect house layout.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        strategies = sorted(strategy_stats.keys())
        vp_earned = [np.mean(strategy_stats[s]['total_vp_earned']) if strategy_stats[s]['total_vp_earned'] else 0
                     for s in strategies]
        vp_spent = [np.mean(strategy_stats[s]['total_vp_spent']) if strategy_stats[s]['total_vp_spent'] else 0
                    for s in strategies]

        fig, ax = plt.subplots(figsize=self.plot_style['figsize_standard'])
        x = np.arange(len(strategies))
        width = 0.35

        ax.barh(x - width/2, vp_earned, width, label='VP Earned',
                color=self.colors['success'], edgecolor='black', alpha=0.8)
        ax.barh(x + width/2, vp_spent, width, label='VP Spent',
                color=self.colors['warning'], edgecolor='black', alpha=0.8)

        ax.set_xlabel('Average VP', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('VP Economics by Strategy', fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.set_yticks(x)
        ax.set_yticklabels(strategies, fontsize=self.plot_style['tick_fontsize'])
        ax.legend(fontsize=self.plot_style['legend_fontsize'])
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=self.plot_style['tick_fontsize'])

        img_path = f"{self.temp_dir}/vp_earned_vs_spent.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=4*inch))
        self.story.append(Spacer(1, 0.3*inch))

        # Works Played vs Discarded
        subtitle = Paragraph("Works Played vs Cards Discarded", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This chart compares the average number of works successfully placed (blue bars) versus "
            "cards discarded (red bars) for each strategy. The ratio reveals card management efficiency.<br/><br/>"
            "<b>How to interpret:</b> A higher ratio of works played to cards discarded indicates better hand management "
            "and strategic planning. High discard rates may suggest poor artist selection, inadequate room preparation, or "
            "overly aggressive commission targeting. The ideal strategy places most cards it draws.<br/><br/>"
            "<b>Why it matters:</b> Every discarded card represents a missed opportunity for points. Efficient strategies "
            "minimize waste by maintaining appropriate room tiles and selecting artists that match their hand composition.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.15*inch))

        works_played = [np.mean(strategy_stats[s]['total_works_played']) if strategy_stats[s]['total_works_played'] else 0
                        for s in strategies]
        cards_discarded = [np.mean(strategy_stats[s]['total_cards_discarded']) if strategy_stats[s]['total_cards_discarded'] else 0
                           for s in strategies]

        fig, ax = plt.subplots(figsize=self.plot_style['figsize_standard'])
        ax.barh(x - width/2, works_played, width, label='Works Played',
                color=self.colors['primary'], edgecolor='black', alpha=0.8)
        ax.barh(x + width/2, cards_discarded, width, label='Cards Discarded',
                color=self.colors['danger'], edgecolor='black', alpha=0.8)

        ax.set_xlabel('Average Cards', fontsize=self.plot_style['label_fontsize'])
        ax.set_title('Card Usage by Strategy', fontsize=self.plot_style['title_fontsize'], pad=10)
        ax.set_yticks(x)
        ax.set_yticklabels(strategies, fontsize=self.plot_style['tick_fontsize'])
        ax.legend(fontsize=self.plot_style['legend_fontsize'])
        ax.grid(axis='x', alpha=0.3)
        ax.tick_params(axis='x', labelsize=self.plot_style['tick_fontsize'])

        img_path = f"{self.temp_dir}/works_vs_discarded.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=4*inch))
        self.story.append(Spacer(1, 0.3*inch))

        note = Paragraph(
            "<i>Note: Detailed tabular breakdowns of works by art type and theme are available in the Annexes section.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)
        self.story.append(Spacer(1, 0.2*inch))

    def _add_advanced_mode_statistics(self):
        """Add advanced mode specific statistics (room tiles, artist/theme usage)."""
        # Check if this is advanced mode data
        has_room_tiles = 'p1_room_tiles' in self.df.columns
        if not has_room_tiles:
            return  # Skip this section for base game

        self.story.append(PageBreak())

        title = Paragraph("PART 5: Advanced Mode Statistics", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Advanced mode introduces room tiles and advantage cards, adding strategic depth to the base game. "
            "This section analyzes how strategies utilize these advanced features: tile acquisition patterns, "
            "advantage card preferences, and the impact on overall performance.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Room Tile Acquisition Stats
        subtitle = Paragraph("Room Tile Acquisition", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This table displays room tile acquisition statistics: average tiles acquired per game, "
            "minimum and maximum values, and standard deviation. Room tiles cost 2 VP each but are necessary to place works.<br/><br/>"
            "<b>How to interpret:</b> Higher average tile counts suggest more aggressive expansion strategies, while lower "
            "counts indicate conservative play. High standard deviation means inconsistent acquisition (adapting to game state), "
            "while low deviation suggests fixed tile-buying patterns.<br/><br/>"
            "<b>Why it matters:</b> Tile acquisition represents a strategic investment trade-off: spending VP now to enable "
            "future point generation. Optimal strategies balance tile costs against placement needs and VP efficiency.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate room tile stats by strategy and player count
        tile_stats_by_player_count = {2: {}, 3: {}, 4: {}, 'Overall': {}}

        for idx, row in self.df.iterrows():
            # Determine player count for this game
            player_count = sum(1 for pos in range(1, 5)
                             if f'p{pos}_strategy' in self.df.columns and not pd.isna(row[f'p{pos}_strategy']))

            if player_count not in [2, 3, 4]:
                continue

            for pos in range(1, player_count + 1):
                strategy_col = f'p{pos}_strategy'
                tiles_col = f'p{pos}_room_tiles'

                if strategy_col not in self.df.columns or tiles_col not in self.df.columns:
                    continue

                strategy = row[strategy_col]
                tiles = row[tiles_col]

                if pd.isna(strategy) or pd.isna(tiles):
                    continue

                # Add to player-count-specific stats
                if strategy not in tile_stats_by_player_count[player_count]:
                    tile_stats_by_player_count[player_count][strategy] = []
                tile_stats_by_player_count[player_count][strategy].append(tiles)

                # Add to overall stats
                if strategy not in tile_stats_by_player_count['Overall']:
                    tile_stats_by_player_count['Overall'][strategy] = []
                tile_stats_by_player_count['Overall'][strategy].append(tiles)

        if tile_stats_by_player_count['Overall']:
            # Create 4 subplots: 2P, 3P, 4P, Overall
            fig, axes = plt.subplots(2, 2, figsize=self.plot_style['figsize_wide'])
            fig.suptitle('Average Room Tiles Acquired by Strategy and Player Count',
                        fontsize=self.plot_style['title_fontsize'], fontweight='bold')

            plot_configs = [
                (axes[0, 0], 2, '2-Player Games', self.colors['player2']),
                (axes[0, 1], 3, '3-Player Games', self.colors['player3']),
                (axes[1, 0], 4, '4-Player Games', self.colors['player4']),
                (axes[1, 1], 'Overall', 'All Games', self.colors['primary'])
            ]

            for ax, key, title, color in plot_configs:
                if tile_stats_by_player_count[key]:
                    strategies = sorted(tile_stats_by_player_count[key].keys())
                    avgs = [np.mean(tile_stats_by_player_count[key][s]) for s in strategies]

                    bars = ax.bar(range(len(strategies)), avgs, color=color, edgecolor='black', alpha=0.8)
                    ax.set_xticks(range(len(strategies)))
                    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
                    ax.set_ylabel('Avg Tiles', fontsize=self.plot_style['label_fontsize'])
                    ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')
                    ax.grid(axis='y', alpha=0.3)

                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}',
                               ha='center', va='bottom', fontsize=7)
                else:
                    ax.text(0.5, 0.5, f'No {key}P data', ha='center', va='center',
                           transform=ax.transAxes, fontsize=10)
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.tight_layout()
            img_path = os.path.join(self.temp_dir, 'room_tiles_by_player_count.png')
            plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
            plt.close()

            self.story.append(Image(img_path, width=6.5*inch, height=5*inch))
            self.story.append(Spacer(1, 0.2*inch))

            note = Paragraph(
                "<i>Note: Detailed room tile statistics are available in Annex D of the Annexes section.</i>",
                self.styles['BodyText']
            )
            self.story.append(note)
            self.story.append(Spacer(1, 0.3*inch))

        # Advantage Card Selection Statistics
        subtitle = Paragraph("Advantage Card Selection Statistics", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.1*inch))

        explanation = Paragraph(
            "<b>What this shows:</b> This heatmap and accompanying table show which advantage cards each strategy selects "
            "most frequently. Warmer colors (orange/red) indicate higher selection rates, while cooler colors indicate rare selection.<br/><br/>"
            "<b>How to interpret:</b> Card preferences reveal strategic priorities. Universal Exhibition and Patronage provide "
            "flexible VP gains, while cards like Remodeling or Reform enable tactical repositioning. Strategies with diverse "
            "card selections adapt to game state, while focused selections suggest specific tactical approaches.<br/><br/>"
            "<b>Why it matters:</b> Advantage cards provide crucial swing opportunities. Understanding which cards each strategy "
            "values helps predict their decision-making and identify potential exploitable patterns.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
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

            # Create heatmap visualization
            strategies = sorted(advantage_selections.keys())
            cards = sorted(all_cards)

            # Build matrix for heatmap
            data_matrix = []
            for card in cards:
                row = []
                for strategy in strategies:
                    count = advantage_selections[strategy].get(card, 0)
                    total = total_selections_by_strategy[strategy]
                    pct = (count / total * 100) if total > 0 else 0
                    row.append(pct)
                data_matrix.append(row)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(data_matrix, aspect='auto', cmap='YlOrRd')

            # Set ticks
            ax.set_xticks(np.arange(len(strategies)))
            ax.set_yticks(np.arange(len(cards)))
            ax.set_xticklabels(strategies, rotation=45, ha='right')
            ax.set_yticklabels(cards)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Selection Rate (%)', rotation=270, labelpad=20)

            # Add text annotations
            for i in range(len(cards)):
                for j in range(len(strategies)):
                    text = ax.text(j, i, f'{data_matrix[i][j]:.1f}%',
                                  ha="center", va="center", color="black", fontsize=8)

            ax.set_title('Advantage Card Selection Preferences (% of total selections)')
            plt.tight_layout()

            img_path = f"{self.temp_dir}/advantage_cards_heatmap.png"
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close()

            self.story.append(Image(img_path, width=6.5*inch, height=4.5*inch))
            self.story.append(Spacer(1, 0.3*inch))

            note = Paragraph(
                "<i>Note: Detailed advantage card selection frequencies are available in Annex D.</i>",
                self.styles['BodyText']
            )
            self.story.append(note)
            self.story.append(Spacer(1, 0.2*inch))

        # Artist/Theme Usage (if available in logs)
        # Note: This would require parsing log files or adding columns to CSV
        # For now, add placeholder text
        subtitle = Paragraph("Artist and Theme Analysis", self.styles['Heading2'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        analysis_note = Paragraph(
            "Analysis of artist types and themes commissioned throughout games. "
            "This data reveals strategic preferences and meta-game trends. "
            "Detailed statistics are available in Annex D.",
            self.styles['BodyText']
        )
        self.story.append(analysis_note)
        self.story.append(Spacer(1, 0.2*inch))

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

            # Create visualization for this player count
            if matchups:
                # Collect all matchup data for visualization
                viz_data = []
                for strategy in sorted(matchups.keys()):
                    # Get all opponent combinations for this strategy
                    for opponents, record in matchups[strategy].items():
                        wins = record['wins']
                        games = record['games']
                        if games >= 10:  # Only show matchups with significant sample size
                            win_rate = (wins / games * 100) if games > 0 else 0
                            opponent_str = ' + '.join(opponents)
                            if len(opponent_str) > 30:
                                opponent_str = opponent_str[:27] + '...'
                            viz_data.append({
                                'strategy': strategy,
                                'opponents': opponent_str,
                                'label': f"{strategy} vs {opponent_str}",
                                'win_rate': win_rate,
                                'games': games
                            })

                # Sort by win rate and get top 15 most interesting matchups
                viz_data.sort(key=lambda x: x['win_rate'], reverse=True)
                top_viz = viz_data[:15]

                if top_viz:
                    # Create horizontal bar chart
                    fig, ax = plt.subplots(figsize=(10, max(6, len(top_viz) * 0.4)))

                    labels = [item['label'] for item in top_viz]
                    win_rates = [item['win_rate'] for item in top_viz]
                    games_counts = [item['games'] for item in top_viz]

                    # Color bars based on win rate
                    colors_list = []
                    for wr in win_rates:
                        if wr >= 60:
                            colors_list.append(self.colors['success'])
                        elif wr >= 40:
                            colors_list.append(self.colors['primary'])
                        else:
                            colors_list.append(self.colors['danger'])

                    bars = ax.barh(range(len(labels)), win_rates, color=colors_list, edgecolor='black', alpha=0.8)
                    ax.set_yticks(range(len(labels)))
                    ax.set_yticklabels(labels, fontsize=8)
                    ax.set_xlabel('Win Rate (%)', fontsize=self.plot_style['label_fontsize'])
                    ax.set_title(f'Top Matchups: {player_count}-Player Games (min 10 games)',
                                fontsize=self.plot_style['title_fontsize'], fontweight='bold')
                    ax.axvline(50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
                    ax.set_xlim(0, 100)
                    ax.grid(axis='x', alpha=0.3)
                    ax.legend(fontsize=self.plot_style['legend_fontsize'])

                    # Add value labels
                    for i, (bar, games) in enumerate(zip(bars, games_counts)):
                        width = bar.get_width()
                        ax.text(width + 2, bar.get_y() + bar.get_height()/2.,
                               f'{width:.1f}% (n={games})',
                               ha='left', va='center', fontsize=7)

                    plt.tight_layout()
                    img_path = os.path.join(self.temp_dir, f'matchups_{player_count}p.png')
                    plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
                    plt.close()

                    self.story.append(Image(img_path, width=6.5*inch, height=max(4*inch, len(top_viz) * 0.25*inch)))
                    self.story.append(Spacer(1, 0.2*inch))
                else:
                    note = Paragraph(
                        f"<i>Not enough matchup data with sufficient sample sizes for {player_count}-player games.</i>",
                        self.styles['BodyText']
                    )
                    self.story.append(note)
                    self.story.append(Spacer(1, 0.2*inch))

            # Add summary stats for this player count
            total_games = len(games_subset)
            avg_score = games_subset['winner_score'].mean() if 'winner_score' in games_subset.columns else 0

            summary = Paragraph(
                f"<i>Total {player_count}P games: {total_games:,} | Avg winning score: {avg_score:.1f} VP</i>",
                self.styles['BodyText']
            )
            self.story.append(summary)
            self.story.append(Spacer(1, 0.3*inch))

    def _add_position_analysis(self):
        """Add position-based performance analysis segregated by player count."""
        self.story.append(Paragraph("Position-Based Performance", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section analyzes how each strategy performs from different starting positions, "
            "segregated by player count. Position effects vary significantly: in 2P games only positions "
            "1-2 exist, in 3P games positions 1-3, and in 4P games all positions 1-4 are relevant.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Get all strategies
        strategies = sorted(self.df['winner_strategy'].dropna().unique())

        # Calculate position win rates by player count
        position_data_by_player_count = {2: {}, 3: {}, 4: {}}

        for player_count in [2, 3, 4]:
            # Filter games by player count
            games_mask = self.df.apply(
                lambda row: sum(1 for pos in range(1, 5)
                              if f'p{pos}_strategy' in self.df.columns
                              and not pd.isna(row[f'p{pos}_strategy'])) == player_count,
                axis=1
            )
            games_subset = self.df[games_mask]

            if len(games_subset) == 0:
                continue

            position_data = {strategy: [0] * player_count for strategy in strategies}

            for pos in range(1, player_count + 1):
                strategy_col = f'p{pos}_strategy'
                for strategy in strategies:
                    mask = (games_subset[strategy_col] == strategy) & (games_subset['winner_position'] == pos)
                    wins = mask.sum()
                    total = (games_subset[strategy_col] == strategy).sum()
                    win_rate = (wins / total * 100) if total > 0 else 0
                    position_data[strategy][pos - 1] = win_rate

            position_data_by_player_count[player_count] = position_data

        # Create 3 heatmaps: 2P, 3P, 4P
        fig, axes = plt.subplots(1, 3, figsize=(14, 6))
        fig.suptitle('Strategy Win Rate by Starting Position and Player Count',
                    fontsize=self.plot_style['title_fontsize'], fontweight='bold')

        plot_configs = [
            (axes[0], 2, '2-Player Games', ['P1', 'P2']),
            (axes[1], 3, '3-Player Games', ['P1', 'P2', 'P3']),
            (axes[2], 4, '4-Player Games', ['P1', 'P2', 'P3', 'P4'])
        ]

        for ax, player_count, title, position_labels in plot_configs:
            if position_data_by_player_count[player_count]:
                position_data = position_data_by_player_count[player_count]
                data_matrix = np.array([position_data[s] for s in strategies])

                im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=50)

                ax.set_xticks(np.arange(len(position_labels)))
                ax.set_yticks(np.arange(len(strategies)))
                ax.set_xticklabels(position_labels, fontsize=8)
                ax.set_yticklabels(strategies, fontsize=7)
                ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')

                # Add text annotations
                for i in range(len(strategies)):
                    for j in range(len(position_labels)):
                        ax.text(j, i, f'{data_matrix[i, j]:.0f}',
                               ha="center", va="center", color="black", fontsize=6)
            else:
                ax.text(0.5, 0.5, f'No {player_count}P data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        # Add single colorbar for all plots
        cbar = fig.colorbar(im, ax=axes, label='Win Rate (%)', fraction=0.046, pad=0.04)

        plt.tight_layout()
        img_path = os.path.join(self.temp_dir, 'position_heatmap_by_player_count.png')
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=3.5*inch))
        self.story.append(Spacer(1, 0.2*inch))

        explanation = Paragraph(
            "<b>Interpretation:</b> Expected win rates are 50% in 2P (positions 1-2), 33.3% in 3P (positions 1-3), "
            "and 25% in 4P (positions 1-4). Values significantly above these baselines indicate positional advantage.",
            self.styles['PlotExplanation']
        )
        self.story.append(explanation)
        self.story.append(PageBreak())

    def _add_vp_analysis(self):
        """Add VP distribution analysis segregated by player count."""
        self.story.append(Paragraph("Victory Point Analysis", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Victory Point (VP) distribution by strategy, segregated by 2-player, 3-player, 4-player, and overall games. "
            "Box plots show median, quartiles, and outliers for each strategy.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Gather scores by strategy and player count
        strategy_scores_by_player_count = {2: {}, 3: {}, 4: {}, 'Overall': {}}

        for idx, row in self.df.iterrows():
            # Determine player count for this game
            player_count = sum(1 for pos in range(1, 5)
                             if f'p{pos}_strategy' in self.df.columns and not pd.isna(row[f'p{pos}_strategy']))

            if player_count not in [2, 3, 4]:
                continue

            for pos in range(1, player_count + 1):
                strategy_col = f'p{pos}_strategy'
                score_col = f'p{pos}_final_score'

                if strategy_col not in self.df.columns or score_col not in self.df.columns:
                    continue

                strategy = row[strategy_col]
                score = row[score_col]

                if pd.isna(strategy) or pd.isna(score):
                    continue

                # Add to player-count-specific scores
                if strategy not in strategy_scores_by_player_count[player_count]:
                    strategy_scores_by_player_count[player_count][strategy] = []
                strategy_scores_by_player_count[player_count][strategy].append(score)

                # Add to overall scores
                if strategy not in strategy_scores_by_player_count['Overall']:
                    strategy_scores_by_player_count['Overall'][strategy] = []
                strategy_scores_by_player_count['Overall'][strategy].append(score)

        # Create 4 box plots: 2P, 3P, 4P, Overall
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('VP Distribution by Strategy and Player Count',
                    fontsize=self.plot_style['title_fontsize'] + 2, fontweight='bold')

        plot_configs = [
            (axes[0, 0], 2, '2-Player Games'),
            (axes[0, 1], 3, '3-Player Games'),
            (axes[1, 0], 4, '4-Player Games'),
            (axes[1, 1], 'Overall', 'All Games')
        ]

        for ax, key, title in plot_configs:
            if strategy_scores_by_player_count[key]:
                strategies = sorted(strategy_scores_by_player_count[key].keys())
                data_to_plot = [strategy_scores_by_player_count[key][s] for s in strategies]

                bp = ax.boxplot(data_to_plot, labels=strategies, patch_artist=True)

                # Color boxes using unified colors
                colors_list = [self.colors['primary'], self.colors['success'],
                              self.colors['warning'], self.colors['danger']]
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors_list[i % len(colors_list)])
                    patch.set_alpha(0.7)

                ax.set_ylabel('Victory Points', fontsize=self.plot_style['label_fontsize'])
                ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45, labelsize=8)

                # Make x-axis labels readable
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
            else:
                ax.text(0.5, 0.5, f'No {key}P data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        img_path = os.path.join(self.temp_dir, 'vp_distribution_by_player_count.png')
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=6*inch))
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: Detailed VP statistics (mean, std dev, min, max, median) are available in the Annexes section.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(PageBreak())

    def _add_score_difference_analysis(self):
        """Add score difference analysis segregated by player count."""
        self.story.append(Paragraph("Score Differential Analysis", self.styles['CustomHeading']))
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Score differential measures the gap between the winner and other players, "
            "indicating game competitiveness and strategy dominance. Analysis is segregated "
            "by 2-player, 3-player, 4-player, and overall games.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.2*inch))

        # Collect score differences by player count
        score_diff_by_player_count = {2: [], 3: [], 4: [], 'Overall': []}

        for idx, row in self.df.iterrows():
            # Determine player count for this game
            player_count = sum(1 for pos in range(1, 5)
                             if f'p{pos}_strategy' in self.df.columns and not pd.isna(row[f'p{pos}_strategy']))

            if player_count not in [2, 3, 4]:
                continue

            score_diff = row.get('score_difference')
            if pd.isna(score_diff):
                continue

            score_diff_by_player_count[player_count].append(score_diff)
            score_diff_by_player_count['Overall'].append(score_diff)

        # Create 4 histograms: 2P, 3P, 4P, Overall
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Score Differential Distribution by Player Count',
                    fontsize=self.plot_style['title_fontsize'] + 2, fontweight='bold')

        plot_configs = [
            (axes[0, 0], 2, '2-Player Games', self.colors['player2']),
            (axes[0, 1], 3, '3-Player Games', self.colors['player3']),
            (axes[1, 0], 4, '4-Player Games', self.colors['player4']),
            (axes[1, 1], 'Overall', 'All Games', self.colors['primary'])
        ]

        for ax, key, title, color in plot_configs:
            if score_diff_by_player_count[key]:
                data = score_diff_by_player_count[key]
                mean_val = np.mean(data)
                median_val = np.median(data)

                ax.hist(data, bins=20, color=color, edgecolor='black', alpha=0.7)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {mean_val:.1f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                          label=f'Median: {median_val:.1f}')

                ax.set_xlabel('Score Difference (Winner - Last Place)', fontsize=self.plot_style['label_fontsize'])
                ax.set_ylabel('Frequency', fontsize=self.plot_style['label_fontsize'])
                ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')
                ax.legend(fontsize=self.plot_style['legend_fontsize'])
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'No {key}P data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        img_path = os.path.join(self.temp_dir, 'score_diff_by_player_count.png')
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=6*inch))
        self.story.append(Spacer(1, 0.2*inch))

        # Summary statistics for each player count
        summary_text = "<b>Score Differential Statistics:</b><br/><br/>"
        for key in [2, 3, 4, 'Overall']:
            if score_diff_by_player_count[key]:
                data = score_diff_by_player_count[key]
                label = f"{key}-Player" if isinstance(key, int) else "Overall"
                summary_text += (
                    f"<b>{label}:</b> Mean={np.mean(data):.1f} VP, "
                    f"Median={np.median(data):.1f} VP, "
                    f"StdDev={np.std(data):.1f} VP<br/>"
                )

        self.story.append(Paragraph(summary_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.2*inch))
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

        # Calculate strategy usage by player count
        strategy_usage_by_player_count = {2: {}, 3: {}, 4: {}, 'Overall': {}}

        for idx, row in self.df.iterrows():
            # Determine player count for this game
            player_count = sum(1 for pos in range(1, 5)
                             if f'p{pos}_strategy' in self.df.columns and not pd.isna(row[f'p{pos}_strategy']))

            if player_count not in [2, 3, 4]:
                continue

            for pos in range(1, player_count + 1):
                strategy_col = f'p{pos}_strategy'
                if strategy_col not in self.df.columns:
                    continue

                strategy = row[strategy_col]
                if pd.isna(strategy):
                    continue

                # Add to player-count-specific usage
                if strategy not in strategy_usage_by_player_count[player_count]:
                    strategy_usage_by_player_count[player_count][strategy] = 0
                strategy_usage_by_player_count[player_count][strategy] += 1

                # Add to overall usage
                if strategy not in strategy_usage_by_player_count['Overall']:
                    strategy_usage_by_player_count['Overall'][strategy] = 0
                strategy_usage_by_player_count['Overall'][strategy] += 1

        # Create 4 bar charts: 2P, 3P, 4P, Overall
        fig, axes = plt.subplots(2, 2, figsize=self.plot_style['figsize_wide'])
        fig.suptitle('Strategy Usage Frequency by Player Count',
                    fontsize=self.plot_style['title_fontsize'], fontweight='bold')

        plot_configs = [
            (axes[0, 0], 2, '2-Player Games', self.colors['player2']),
            (axes[0, 1], 3, '3-Player Games', self.colors['player3']),
            (axes[1, 0], 4, '4-Player Games', self.colors['player4']),
            (axes[1, 1], 'Overall', 'All Games', self.colors['primary'])
        ]

        for ax, key, title, color in plot_configs:
            if strategy_usage_by_player_count[key]:
                strategies = sorted(strategy_usage_by_player_count[key].keys())
                counts = [strategy_usage_by_player_count[key][s] for s in strategies]

                bars = ax.bar(range(len(strategies)), counts, color=color, edgecolor='black', alpha=0.8)
                ax.set_xticks(range(len(strategies)))
                ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Usage Count', fontsize=self.plot_style['label_fontsize'])
                ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')
                ax.grid(axis='y', alpha=0.3)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=7)
            else:
                ax.text(0.5, 0.5, f'No {key}P data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        img_path = os.path.join(self.temp_dir, 'strategy_usage_by_player_count.png')
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=5*inch))
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: In balanced simulations, each strategy appears equally across all positions. "
            "Detailed usage statistics by position are available in the Annexes section.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)
        self.story.append(Spacer(1, 0.2*inch))

    def _add_head_to_head_analysis(self):
        """Add head-to-head win rate matrix segregated by player count."""
        self.story.append(PageBreak())

        title = Paragraph("Head-to-Head Win Rates", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "Head-to-head win rate matrices showing performance when Strategy A (row) plays against "
            "Strategy B (column), segregated by 2-player, 3-player, 4-player, and overall games. "
            "Values represent the percentage of games won by the row strategy.",
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
                if not pd.isna(strategy):
                    strategy_usage[strategy] = True

        all_strat_names = sorted(strategy_usage.keys())

        # Build head-to-head matrices by player count
        h2h_by_player_count = {2: {}, 3: {}, 4: {}, 'Overall': {}}
        for key in h2h_by_player_count:
            h2h_by_player_count[key] = {
                s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names}
                for s1 in all_strat_names
            }

        # Calculate head-to-head stats by player count
        for idx, row in self.df.iterrows():
            # Determine player count for this game
            player_count = sum(1 for pos in range(1, 5)
                             if f'p{pos}_strategy' in self.df.columns and not pd.isna(row[f'p{pos}_strategy']))

            if player_count not in [2, 3, 4]:
                continue

            winner_strategy = row['winner_strategy']
            winner_pos = row['winner_position']

            if pd.isna(winner_strategy):
                continue

            # Collect strategies in this game
            strategies = []
            for pos in range(1, player_count + 1):
                col = f'p{pos}_strategy'
                if col in self.df.columns and not pd.isna(row[col]):
                    strategies.append((pos, row[col]))

            for pos, opponent in strategies:
                if pos != winner_pos:
                    # Add to player-count-specific matrix
                    h2h_by_player_count[player_count][winner_strategy][opponent]['wins'] += 1
                    h2h_by_player_count[player_count][winner_strategy][opponent]['games'] += 1
                    h2h_by_player_count[player_count][opponent][winner_strategy]['games'] += 1

                    # Add to overall matrix
                    h2h_by_player_count['Overall'][winner_strategy][opponent]['wins'] += 1
                    h2h_by_player_count['Overall'][winner_strategy][opponent]['games'] += 1
                    h2h_by_player_count['Overall'][opponent][winner_strategy]['games'] += 1

        # Create 4 heatmaps: 2P, 3P, 4P, Overall
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Head-to-Head Win Rate Matrices by Player Count',
                    fontsize=self.plot_style['title_fontsize'] + 2, fontweight='bold')

        plot_configs = [
            (axes[0, 0], 2, '2-Player Games'),
            (axes[0, 1], 3, '3-Player Games'),
            (axes[1, 0], 4, '4-Player Games'),
            (axes[1, 1], 'Overall', 'All Games')
        ]

        for ax, key, title in plot_configs:
            h2h_matrix = h2h_by_player_count[key]

            # Build matrix data
            matrix_data = []
            for s1 in all_strat_names:
                row_data = []
                for s2 in all_strat_names:
                    if s1 == s2:
                        row_data.append(np.nan)
                    else:
                        games = h2h_matrix[s1][s2]['games']
                        if games >= 5:  # Minimum threshold for meaningful win rate
                            wins = h2h_matrix[s1][s2]['wins']
                            win_rate = wins / games * 100
                            row_data.append(win_rate)
                        else:
                            row_data.append(np.nan)
                matrix_data.append(row_data)

            # Create heatmap
            im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

            # Labels
            ax.set_xticks(np.arange(len(all_strat_names)))
            ax.set_yticks(np.arange(len(all_strat_names)))
            ax.set_xticklabels([s[:8] for s in all_strat_names], rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels([s[:12] for s in all_strat_names], fontsize=7)

            # Add text annotations
            for i in range(len(all_strat_names)):
                for j in range(len(all_strat_names)):
                    if not np.isnan(matrix_data[i][j]):
                        ax.text(j, i, f"{matrix_data[i][j]:.0f}",
                               ha="center", va="center", color="black", fontsize=6)

            ax.set_title(title, fontsize=self.plot_style['label_fontsize'], fontweight='bold')

        # Add single colorbar for all plots
        fig.colorbar(im, ax=axes, label='Win Rate %', fraction=0.046, pad=0.04)

        plt.tight_layout()
        img_path = os.path.join(self.temp_dir, 'h2h_matrix_by_player_count.png')
        plt.savefig(img_path, dpi=self.plot_style['dpi'], bbox_inches='tight')
        plt.close()

        self.story.append(Image(img_path, width=6.5*inch, height=6*inch))
        self.story.append(Spacer(1, 0.2*inch))

        note = Paragraph(
            "<i>Note: Only matchups with at least 5 games are shown to ensure statistical relevance. "
            "Diagonal cells (same strategy) are left blank.</i>",
            self.styles['BodyText']
        )
        self.story.append(note)
        self.story.append(Spacer(1, 0.2*inch))

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

    def _add_annexes(self):
        """Add annexes section with all tabular data."""
        self.story.append(PageBreak())

        title = Paragraph("ANNEXES: Detailed Tabular Data", self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.3*inch))

        intro = Paragraph(
            "This section contains comprehensive tabular data referenced throughout the report. "
            "While the main report emphasizes visual communication through charts and graphs, "
            "these tables provide precise numerical values for detailed analysis.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.5*inch))

        # Annex A: Strategy Performance Summary
        self._add_annex_strategy_summary()

        # Annex B: Player Count Statistics
        self._add_annex_player_count_stats()

        # Annex C: Works & VP Detailed Breakdown
        self._add_annex_works_vp_breakdown()

        # Annex D: Advanced Mode Tables (if applicable)
        self._add_annex_advanced_mode()

    def _add_annex_strategy_summary(self):
        """Annex A: Strategy Performance Summary Table."""
        self.story.append(PageBreak())

        subtitle = Paragraph("Annex A: Strategy Performance Summary", self.styles['CustomHeading'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        # Calculate strategy statistics
        strategy_stats = {}
        for pos in range(1, 5):
            strategy_col = f'p{pos}_strategy'
            score_col = f'p{pos}_final_score'

            if strategy_col not in self.df.columns or score_col not in self.df.columns:
                continue

            for idx, row in self.df.iterrows():
                strategy = row[strategy_col]
                score = row[score_col]

                if pd.isna(strategy) or pd.isna(score):
                    continue

                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {'wins': 0, 'games': 0, 'total_score': 0}

                strategy_stats[strategy]['games'] += 1
                strategy_stats[strategy]['total_score'] += score

                if row['winner_strategy'] == strategy:
                    strategy_stats[strategy]['wins'] += 1

        strategies = sorted(strategy_stats.keys())
        table_data = [['Strategy', 'Games Played', 'Wins', 'Win Rate', 'Avg Score']]

        for strategy in strategies:
            stats = strategy_stats[strategy]
            win_rate = (stats['wins'] / stats['games'] * 100)
            avg_score = stats['total_score'] / stats['games']
            table_data.append([
                strategy,
                f"{stats['games']:,}",
                f"{stats['wins']:,}",
                f"{win_rate:.1f}%",
                f"{avg_score:.1f}"
            ])

        table = Table(table_data, colWidths=[2.2*inch, 1.1*inch, 0.9*inch, 1*inch, 1*inch])
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

    def _add_annex_player_count_stats(self):
        """Annex B: Player Count Statistics."""
        subtitle = Paragraph("Annex B: Player Count Statistics", self.styles['CustomHeading'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        # Determine player count for each game
        player_count_data = {}
        for idx, row in self.df.iterrows():
            num_players = sum(1 for pos in range(1, 5)
                            if f'p{pos}_strategy' in self.df.columns
                            and not pd.isna(row[f'p{pos}_strategy']))

            if num_players not in player_count_data:
                player_count_data[num_players] = {'games': 0, 'strategies': {}}

            player_count_data[num_players]['games'] += 1

            winner = row['winner_strategy']
            if not pd.isna(winner):
                if winner not in player_count_data[num_players]['strategies']:
                    player_count_data[num_players]['strategies'][winner] = 0
                player_count_data[num_players]['strategies'][winner] += 1

        # Summary table
        table_data = [['Player Count', 'Total Games', '% of All Games', 'Avg Winning Score', 'Avg Score Diff']]

        for player_count in sorted(player_count_data.keys()):
            data = player_count_data[player_count]
            total_games = data['games']
            pct = (total_games / len(self.df)) * 100

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

        table = Table(table_data, colWidths=[1.5*inch, 1.2*inch, 1.3*inch, 1.5*inch, 1.3*inch])
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

        # Strategy performance by player count table
        subtitle2 = Paragraph("Strategy Win Rates by Player Count", self.styles['Heading2'])
        self.story.append(subtitle2)
        self.story.append(Spacer(1, 0.1*inch))

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
        self.story.append(Spacer(1, 0.3*inch))

    def _add_annex_works_vp_breakdown(self):
        """Annex C: Works & VP Detailed Breakdown."""
        self.story.append(PageBreak())

        subtitle = Paragraph("Annex C: Works & VP Detailed Breakdown", self.styles['CustomHeading'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

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
        subtitle2 = Paragraph("Works Played by Art Type", self.styles['Heading2'])
        self.story.append(subtitle2)
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
        subtitle3 = Paragraph("Works Played by Theme", self.styles['Heading2'])
        self.story.append(subtitle3)
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
        self.story.append(Spacer(1, 0.3*inch))

    def _add_annex_advanced_mode(self):
        """Annex D: Advanced Mode Tables (if applicable)."""
        has_room_tiles = 'p1_room_tiles' in self.df.columns
        if not has_room_tiles:
            return

        self.story.append(PageBreak())

        subtitle = Paragraph("Annex D: Advanced Mode Statistics", self.styles['CustomHeading'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.2*inch))

        # Room Tile Acquisition Table
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
            subtitle2 = Paragraph("Room Tile Acquisition Statistics", self.styles['Heading2'])
            self.story.append(subtitle2)
            self.story.append(Spacer(1, 0.2*inch))

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
