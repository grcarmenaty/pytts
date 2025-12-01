# PyTTS - Python Tabletop Simulator

A comprehensive simulation framework for the Modernisme board game with AI strategy analysis.

## Features

- 🎲 **Full Game Implementation** - Complete Modernisme rules (v0.4.1)
- 🤖 **7 AI Strategies** - From random baseline to sophisticated optimizers
- ⚡ **Multiprocessing** - Parallel game execution for fast simulations
- 📊 **Comprehensive Analytics** - 210+ data points per game
- 📈 **Beautiful Reports** - Automatic PDF generation with visualizations
- 📝 **Detailed Logging** - Turn-by-turn game transcripts

## Quick Start

### Installation with uv (Recommended)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt

# Or install the project
uv pip install -e .
```

### Installation with pip

```bash
pip install -r requirements.txt
```

### Run Simulations

```bash
# Run 1,000 games for testing
python examples/run_simulations.py 1000

# Run 100,000 games (default) with all CPU cores
python examples/run_simulations.py

# Run with specific number of processes
python examples/run_simulations.py 10000 4
```

## Output Files

All results are saved to `examples/modernisme_logs/`:

- `game_XXXXXX_[timestamp].log` - Human-readable game transcript
- `game_XXXXXX_[timestamp].csv` - Structured game data (210+ columns)
- `all_games_[timestamp].csv` - Aggregated data from all games
- `simulation_report_[timestamp].pdf` - Professional analysis report
- `simulation_summary.txt` - Text-based statistics summary

## Project Structure

```
pytts/
├── pytts/                      # Core game framework
│   ├── game.py                 # Base game classes
│   ├── player.py               # Player management
│   ├── card.py                 # Card system
│   ├── deck.py                 # Deck management
│   └── strategy.py             # AI strategies (7 implementations)
├── examples/
│   ├── modernisme_game.py      # Full Modernisme implementation
│   ├── run_simulations.py      # Simulation runner
│   ├── generate_report.py      # PDF report generation
│   └── modernisme_logs/        # Output directory (gitignored)
├── STRATEGIES.md               # Detailed strategy documentation
├── requirements.txt            # Dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # This file
```

## Strategies

Seven AI strategies are implemented:

1. **Random** - Baseline control (random selection)
2. **Maximum Works** - Quantity focus (play most cards)
3. **High Value Works** - Quality focus (play highest VP cards)
4. **Room Theme/Type Optimizer** - Complex spatial optimization
5. **Theme Fashion Focus** - Targets moda_tema objectives
6. **Set Fashion Focus** - Targets moda_conjunto objectives
7. **Commission Focus** - Optimizes secret encargo objectives

See [STRATEGIES.md](STRATEGIES.md) for detailed explanations.

## Data Analysis

Each game generates 210+ data columns including:

**Per Player:**
- Final scores and placement
- Total cards played/discarded
- Works by type (Crafts, Painting, Sculpture, Relic)
- Works by theme (Nature, Mythology, Society, Orientalism)

**Turn-by-Turn (16 turns × 4 players):**
- VP at start/end of turn
- VP gained/spent
- Cards played/discarded
- Works commissioned by type/theme

**Game Summary:**
- Winner and strategy
- Score statistics
- Objective cards
- Position-based performance

## PDF Reports

Automatically generated reports include:

- Executive summary with key findings
- Strategy performance charts (bar, pie)
- Position-based win rate heatmap
- VP distribution box plots
- Score differential analysis
- Data-driven insights and recommendations

## Performance

- **Speed**: ~180-270 games/second (with multiprocessing)
- **Scale**: 100,000 games in ~6-10 minutes
- **Memory**: Efficient CSV streaming for large datasets

## Requirements

- Python 3.9+
- pandas 2.0+
- numpy 1.23+
- reportlab 4.0+
- matplotlib 3.7+
- seaborn 0.13+

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run a single game (interactive)
python examples/modernisme_game.py

# Run tests
pytest

# Format code
black .
isort .
```

## Use Cases

- **Strategy Development** - Test new AI strategies
- **Game Balance** - Analyze mechanic effectiveness
- **Research** - Study optimal play patterns
- **Benchmarking** - Compare algorithm performance
- **Data Science** - Large-scale game analytics

## Contributing

Contributions welcome! Areas of interest:

- New AI strategies
- Performance optimizations
- Additional visualizations
- Game variants
- Documentation improvements

## License

MIT License - See LICENSE file for details

## Credits

- Game Design: Modernisme by [Publisher]
- Simulation Framework: PyTTS Contributors
- Strategy Implementation: AI research and development

---

**Ready to analyze 100,000 games?** Start with `python examples/run_simulations.py` 🎲📊
