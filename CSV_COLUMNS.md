# CSV Column Specification

This document describes the CSV columns tracked by the Modernisme simulation system. The simulation outputs comprehensive game data that can be analyzed using the PDF report generator.

## Core Columns (Required)

These columns are essential for basic simulation functionality:

### Game Identification
- `game_id` - Unique identifier for each game
- `timestamp` - Game completion timestamp
- `winner_position` - Position (1-4) of the winning player

### Player Strategy
- `p1_strategy`, `p2_strategy`, `p3_strategy`, `p4_strategy` - Strategy name for each player position

### Player Scores
- `p1_score`, `p2_score`, `p3_score`, `p4_score` - Final score for each player
- `p1_rank`, `p2_rank`, `p3_rank`, `p4_rank` - Final rank (1=winner, 4=last place)

## Basic Statistics Columns

These columns provide fundamental game statistics:

### Cards Played
- `p{pos}_cards_played` - Total cards played by player at position {pos}
- `p{pos}_cards_discarded` - Total cards discarded by player at position {pos}

### Victory Points
- `p{pos}_vp_start_turn_{turn}` - VP at start of turn (turns 1-16)
- `p{pos}_vp_end_turn_{turn}` - VP at end of turn (turns 1-16)
- `p{pos}_vp_gained_turn_{turn}` - VP gained during turn
- `p{pos}_vp_spent_turn_{turn}` - VP spent during turn

## Works Placed (Art Type)

Tracks works placed by art type:

- `p{pos}_works_crafts` - Total crafts works placed
- `p{pos}_works_painting` - Total painting works placed
- `p{pos}_works_sculpture` - Total sculpture works placed
- `p{pos}_works_relic` - Total relic works placed

## Works Placed (Theme)

Tracks works placed by theme:

- `p{pos}_works_nature` - Total nature-themed works placed
- `p{pos}_works_mythology` - Total mythology-themed works placed
- `p{pos}_works_society` - Total society-themed works placed
- `p{pos}_works_orientalism` - Total orientalism-themed works placed

## Advanced Tracking Columns (Optional)

These columns enable advanced analytics in the PDF report. They are optional but unlock additional visualization sections.

### Artist Acquisition Sources

Tracks where artists were acquired from:

**Primary column names** (recommended):
- `p{pos}_from_discards` - Artists acquired from discard pile
- `p{pos}_from_drawn` - Artists acquired by drawing new cards
- `p{pos}_from_nearest` - Artists acquired from nearest player's discard

**Alternative column names** (also supported):
- `p{pos}_artists_from_discards`
- `p{pos}_artists_from_drawn`
- `p{pos}_artists_from_nearest_player`
- `p{pos}_cards_from_discards`
- `p{pos}_cards_from_drawn`
- `p{pos}_cards_from_nearest`

**Report Section**: "Artist Acquisition Sources" (Advanced Mode)
- Shows stacked bar chart of acquisition sources by strategy
- Helps identify strategies that leverage discard pile vs drawing

### Artist Selection by Type

Tracks which types of artists were selected/acquired (not just works placed):

**Primary column names** (recommended):
- `p{pos}_artists_crafts` - Crafts artist cards selected
- `p{pos}_artists_painting` - Painting artist cards selected
- `p{pos}_artists_sculpture` - Sculpture artist cards selected
- `p{pos}_artists_relic` - Relic artist cards selected

**Alternative column names** (also supported):
- `p{pos}_artists_selected_crafts`
- `p{pos}_artists_selected_painting`
- `p{pos}_artists_selected_sculpture`
- `p{pos}_artists_selected_relic`
- `p{pos}_artist_type_crafts`
- `p{pos}_artist_type_painting`
- `p{pos}_artist_type_sculpture`
- `p{pos}_artist_type_relic`

**Report Section**: "Artist Type and Theme Selection" (Advanced Mode)
- Shows stacked bar chart of artist type selection by strategy
- Reveals strategic preferences in artist acquisition

### Artist Selection by Theme

Tracks which themes of artists were selected/acquired:

**Primary column names** (recommended):
- `p{pos}_artists_nature` - Nature artist cards selected
- `p{pos}_artists_mythology` - Mythology artist cards selected
- `p{pos}_artists_society` - Society artist cards selected
- `p{pos}_artists_orientalism` - Orientalism artist cards selected

**Alternative column names** (also supported):
- `p{pos}_artists_selected_nature`
- `p{pos}_artists_selected_mythology`
- `p{pos}_artists_selected_society`
- `p{pos}_artists_selected_orientalism`
- `p{pos}_artist_theme_nature`
- `p{pos}_artist_theme_mythology`
- `p{pos}_artist_theme_society`
- `p{pos}_artist_theme_orientalism`

**Report Section**: "Artist Type and Theme Selection" (Advanced Mode)
- Shows stacked bar chart of artist theme selection by strategy
- Helps understand thematic preferences and synergies

## Difference: Works Placed vs Artists Selected

**Important distinction**:

- **Works Placed** (`p{pos}_works_*`) - Tracks which works were actually placed in the exhibition (played to tableau)
- **Artists Selected** (`p{pos}_artists_*`) - Tracks which artist cards were acquired/drawn (may not all be played)

A player might select 15 painting artists but only place 10 painting works. The difference reveals card efficiency, hand management, and strategic adaptation.

## Implementation Notes

### Column Naming Conventions

The report generator supports multiple naming conventions for flexibility:
1. Primary pattern: `p{pos}_artists_{type/theme}`
2. Explicit pattern: `p{pos}_artists_selected_{type/theme}`
3. Alternative pattern: `p{pos}_artist_type_{type}` or `p{pos}_artist_theme_{theme}`

Choose the convention that best fits your simulation code structure.

### Position Numbering

- Player positions are numbered 1-4
- `{pos}` in column names should be replaced with 1, 2, 3, or 4
- Example: `p1_artists_crafts`, `p2_artists_crafts`, etc.

### Data Types

- **Counts** (artists, works, cards) - Integer values (0 or positive)
- **Scores/VP** - Integer values (can be negative for spent VP)
- **Identifiers** - String values
- **Ranks** - Integer values (1-4)

### Missing Data Handling

The report generator gracefully handles missing columns:
- If a column is missing, the corresponding visualization section will show an informative message
- The message explains what columns are needed to enable that analysis
- This allows incremental implementation of tracking features

## Example CSV Structure

```csv
game_id,timestamp,winner_position,p1_strategy,p2_strategy,p1_score,p2_score,p1_works_crafts,p1_artists_crafts,p1_from_discards,...
1,2025-12-04T10:30:00,1,MaximumWorks,HighValueWorks,95,87,5,7,3,...
2,2025-12-04T10:30:05,2,CommissionFocus,ThemeFashion,91,102,3,4,2,...
```

## Adding New Tracking Features

To add new tracking features:

1. Add columns to your simulation's CSV output
2. Update this documentation with column specifications
3. Optionally update `generate_report.py` to create visualizations
4. Test with sample data to verify compatibility

## Report Generation

To generate a PDF report with all available visualizations:

```bash
python examples/generate_report.py examples/modernisme_logs/all_games_[timestamp].csv
```

The report automatically detects which columns are available and generates appropriate visualizations.
