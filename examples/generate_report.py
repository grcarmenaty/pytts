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
        self._add_strategy_analysis()
        self._add_strategy_usage()
        self._add_head_to_head_analysis()
        self._add_matchup_analysis()
        self._add_position_analysis()
        self._add_vp_analysis()
        self._add_score_difference_analysis()
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
            for _, row in self.df.iterrows():
                strategy = row[strategy_col]
                score = row[score_col]
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
            for strategy in self.df[strategy_col]:
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
            for strategy in self.df[f'p{pos}_strategy']:
                strategy_usage[strategy] = True

        all_strat_names = sorted(strategy_usage.keys())
        h2h_matrix = {s1: {s2: {'wins': 0, 'games': 0} for s2 in all_strat_names} for s1 in all_strat_names}

        # Calculate head-to-head stats
        for idx, row in self.df.iterrows():
            winner_strategy = row['winner_strategy']
            winner_pos = row['winner_position']
            strategies = [row[f'p{pos}_strategy'] for pos in range(1, 5)]

            for pos in range(1, 5):
                if pos != winner_pos:
                    opponent = strategies[pos - 1]
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
        """Add matchup performance analysis."""
        self.story.append(PageBreak())

        title = Paragraph("Best and Worst Matchups", self.styles['CustomHeading'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.2*inch))

        intro = Paragraph(
            "This section identifies the best and worst opponent combinations for each strategy, "
            "showing which matchups favor each strategy the most.",
            self.styles['BodyText']
        )
        self.story.append(intro)
        self.story.append(Spacer(1, 0.3*inch))

        # Build matchup performance
        matchup_performance = {}

        for idx, row in self.df.iterrows():
            winner_pos = row['winner_position']
            strategies = tuple(row[f'p{pos}_strategy'] for pos in range(1, 5))

            for pos in range(1, 5):
                strategy = strategies[pos - 1]
                opponents = tuple(strategies[i] for i in range(4) if i != pos - 1)

                if strategy not in matchup_performance:
                    matchup_performance[strategy] = {}
                if opponents not in matchup_performance[strategy]:
                    matchup_performance[strategy][opponents] = {'wins': 0, 'games': 0}

                matchup_performance[strategy][opponents]['games'] += 1
                if pos == winner_pos:
                    matchup_performance[strategy][opponents]['wins'] += 1

        # Display top 3 best and worst for each strategy
        for strategy in sorted(matchup_performance.keys())[:4]:  # Show first 4 strategies to fit
            matchups = matchup_performance[strategy]

            matchup_winrates = []
            for opponents, stats in matchups.items():
                if stats['games'] >= 1:
                    win_rate = stats['wins'] / stats['games'] * 100
                    matchup_winrates.append((opponents, stats['wins'], stats['games'], win_rate))

            matchup_winrates.sort(key=lambda x: x[3], reverse=True)

            # Strategy heading
            strat_title = Paragraph(f"<b>{strategy}</b>", self.styles['BodyText'])
            self.story.append(strat_title)
            self.story.append(Spacer(1, 0.1*inch))

            # Best matchups
            best_text = "<b>Best Matchups:</b><br/>"
            for opponents, wins, games, win_rate in matchup_winrates[:3]:
                opp_str = ', '.join([o[:12] for o in opponents])
                best_text += f"• vs [{opp_str}]: {wins}/{games} ({win_rate:.1f}%)<br/>"

            best_para = Paragraph(best_text, self.styles['BodyText'])
            self.story.append(best_para)
            self.story.append(Spacer(1, 0.1*inch))

            # Worst matchups
            worst_text = "<b>Worst Matchups:</b><br/>"
            for opponents, wins, games, win_rate in matchup_winrates[-3:]:
                opp_str = ', '.join([o[:12] for o in opponents])
                worst_text += f"• vs [{opp_str}]: {wins}/{games} ({win_rate:.1f}%)<br/>"

            worst_para = Paragraph(worst_text, self.styles['BodyText'])
            self.story.append(worst_para)
            self.story.append(Spacer(1, 0.2*inch))


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
