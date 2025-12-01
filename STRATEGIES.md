# Modernisme Strategy Guide

This document describes all AI strategies implemented for the Modernisme game simulation. Each strategy represents a different approach to playing the game, from simple baselines to sophisticated tactical approaches.

---

## Overview

All strategies inherit from the `Strategy` abstract base class and implement the `select_work()` method, which determines which work card to commission from the player's hand on each turn. Strategies can only commission works that match their active artists' types.

---

## Strategy Descriptions

### 1. Random Strategy
**Name:** `"Random"`
**Class:** `RandomStrategy`

#### Description
The baseline strategy that randomly selects any commissionable work from the player's hand.

#### Decision Logic
- Gets all works that can be commissioned (matching active artist types)
- Randomly selects one using `random.choice()`
- No consideration of VP, themes, types, or objectives

#### Use Case
Serves as a control baseline to determine if other strategies perform better than random chance.

#### Strengths
- Unbiased
- Provides true random baseline for comparison

#### Weaknesses
- No strategic planning
- Ignores all game objectives and scoring opportunities

---

### 2. Maximum Works Strategy
**Name:** `"Maximum Works"`
**Class:** `MaxWorksStrategy`

#### Description
Focuses on commissioning as many works as possible by always playing the lowest VP cost works.

#### Decision Logic
- Gets all commissionable works
- Selects the work with the **lowest VP cost**
- Rationale: Lower VP works are easier to commission, allowing more total placements

#### Use Case
Tests if quantity over quality is a viable strategy.

#### Strengths
- Maximizes number of works placed
- May benefit from room completion bonuses (more works = more completed rooms)
- Good for commission objectives requiring specific work counts

#### Weaknesses
- Scores fewer VP per work placed
- May waste high-value cards by discarding them
- Doesn't optimize for objectives or bonuses

---

### 3. High Value Works Strategy
**Name:** `"High Value Works"`
**Class:** `HighValueWorksStrategy`

#### Description
Opposite of Maximum Works - always plays the highest VP value works available.

#### Decision Logic
- Gets all commissionable works
- Selects the work with the **highest VP cost**
- Rationale: Higher VP works score more points immediately

#### Use Case
Tests if quality over quantity wins.

#### Strengths
- Maximizes immediate VP gain per commission
- Works well when hand contains many high-value cards
- May benefit from theme bonuses on expensive works

#### Weaknesses
- Places fewer total works (expensive cards harder to commission)
- May not complete rooms or objectives
- Can be inefficient with VP spending (discarding high-value cards to play other high-value cards)

---

### 4. Room Theme/Type Optimizer Strategy
**Name:** `"Room Theme/Type Optimizer"`
**Class:** `RoomThemeTypeStrategy`

#### Description
The most sophisticated spatial strategy, focusing on completing rooms with same-theme or same-type bonuses.

#### Decision Logic
1. Analyzes current state of all 5 rooms
2. For each commissionable work, calculates a score based on:
   - **Same Type Continuation:** +5.0 if work maintains room's uniform type
   - **Same Theme Continuation:** +5.0 if work maintains room's uniform theme
   - **Room Completion:** +10.0 if work completes a room
   - **First Work in Room:** +2.0 (starting a new room)
   - **Breaking Uniformity:** +0.5 (may enable alternative bonus)
   - **Base VP:** +0.1 × work VP (tiebreaker)
3. Selects work with highest room bonus score

#### Scoring Details
- Complete rooms with all same type: room_size VP
- Complete rooms with all same theme: room_size VP
- Only one bonus per room (type takes priority if both apply)

#### Use Case
Optimizes for end-game room completion bonuses.

#### Strengths
- Excellent at completing rooms systematically
- Maximizes room bonus VP (can be substantial)
- Plans ahead for room completion
- Considers spatial board state

#### Weaknesses
- May ignore other objectives (moda, encargo)
- Doesn't consider theme/type distribution across the entire board
- Can be rigid in room-filling approach

---

### 5. Theme Fashion Focus Strategy
**Name:** `"Theme Fashion Focus"`
**Class:** `ModaTemaStrategy`

#### Description
Targets the public "moda tema" (theme fashion) objective, which awards VP for collecting multiple works of a specific theme.

#### Decision Logic
1. Checks the public moda_tema card for required theme
2. Counts current works of the required theme on player's board
3. **Prioritizes works matching the required theme:**
   - If matching works available: select highest VP matching work
   - If no matching works: select lowest VP work (conserve resources)

#### Moda Tema Scoring
- Typical objective: 3 works of same theme = 3 VP
- Can be claimed multiple times (6 works = 6 VP, etc.)

#### Use Case
Focuses on public objectives that all players compete for.

#### Strengths
- Can consistently score moda tema bonuses
- Simple and focused approach
- Works well when hand contains matching theme cards

#### Weaknesses
- Ignores other scoring opportunities
- May be inefficient if required theme cards don't appear
- Competing players may also target same objective

---

### 6. Set Fashion Focus Strategy
**Name:** `"Set Fashion Focus"`
**Class:** `ModaConjuntoStrategy`

#### Description
Targets the public "moda conjunto" (set fashion) objective, which awards VP for creating adjacent sets of 3 works with specific different themes.

#### Decision Logic
1. Checks the public moda_conjunto card for required theme set (e.g., Nature + Mythology + Society)
2. Counts current works of each required theme
3. **Prioritizes the theme player has least of:**
   - Finds which required theme player needs most
   - Selects highest VP work of that theme
   - If not available, tries other required themes
   - If no matching works: select lowest VP work

#### Moda Conjunto Scoring
- 3 adjacent works with all required themes = 3 VP
- Multiple sets possible if properly positioned
- Requires spatial planning (adjacency matters)

#### Use Case
Optimizes for spatial set collection bonuses.

#### Strengths
- Balances theme distribution for set completion
- Can score multiple set bonuses if positioned well
- Considers both theme diversity and spatial positioning

#### Weaknesses
- Requires specific hand compositions
- Adjacency requirement makes it harder to complete
- May not complete if required themes don't appear together

---

### 7. Commission Focus Strategy
**Name:** `"Commission Focus"`
**Class:** `EncargoStrategy`

#### Description
Focuses exclusively on completing the player's secret commission (encargo) objective.

#### Decision Logic
1. Checks player's secret commission card type
2. For **type_count** objectives (e.g., "2 Relics"):
   - Prioritizes works of the required type
   - Selects highest VP work of that type
3. For **mixed** objectives (e.g., "2 Crafts + 1 Painting"):
   - Calculates progress toward each requirement
   - Focuses on type with least progress
   - Selects highest VP work of needed type
4. For **theme_count** objectives:
   - Similar to type_count approach

#### Encargo Scoring
- Typical value: 3 VP per completed objective
- Secret from other players
- Can complete multiple times (4 Relics = 6 VP with "2 Relics" objective)

#### Use Case
Maximizes secret objective scoring.

#### Strengths
- Laser-focused on guaranteed personal objective
- Works well when commission requirements align with hand
- Can score significantly if objective completed multiple times

#### Weaknesses
- Ignores all other scoring opportunities
- Ineffective if required cards don't appear
- May place works suboptimally for room/spatial bonuses

---

## Strategy Comparison Matrix

| Strategy | Focus | Complexity | Planning Horizon | VP Source Priority |
|----------|-------|------------|------------------|-------------------|
| Random | None | Very Low | None | None |
| Maximum Works | Quantity | Low | Short-term | Work placement quantity |
| High Value Works | Quality | Low | Short-term | Individual work VP |
| Room Theme/Type | Spatial | Very High | Long-term | Room completion bonuses |
| Theme Fashion | Theme collection | Medium | Medium-term | Public theme objectives |
| Set Fashion | Theme diversity | High | Medium-term | Public adjacency objectives |
| Commission Focus | Secret objective | Medium | Long-term | Secret commission objectives |

---

## VP Scoring Breakdown

Understanding how VP is scored helps explain strategy effectiveness:

### Immediate VP (During Play)
- **Base Work VP:** 1-5 VP per work placed
- **Theme Bonus:** +1 VP if artist's theme matches work's theme

### End-Game VP (Final Scoring)
- **Room Completion - Same Type:** room_size VP (2-4 VP per room)
- **Room Completion - Same Theme:** room_size VP (2-4 VP per room)
- **Moda Tema (Theme Fashion):** 3 VP per completion
- **Moda Conjunto (Set Fashion):** 3 VP per adjacent set
- **Encargo (Commission):** 3 VP per completion

---

## Strategy Performance Metrics

When analyzing simulation results, consider:

1. **Win Rate:** Overall games won / total games played
2. **Win Rate by Position:** Performance from different starting positions
3. **Average VP:** Mean final score across all games
4. **VP Standard Deviation:** Consistency of performance
5. **Score Differential:** Gap between winner and losers

---

## Strategy Design Philosophy

### Simple Strategies (Random, Max Works, High Value)
- **Purpose:** Establish baselines
- **Approach:** Single-factor decision making
- **Learning:** Show what doesn't work or minimal approaches

### Tactical Strategies (Theme/Set Fashion, Commission)
- **Purpose:** Optimize for specific objectives
- **Approach:** Target-focused decisions
- **Learning:** Test if focusing on one aspect beats balanced play

### Strategic Strategies (Room Optimizer)
- **Purpose:** Complex multi-factor optimization
- **Approach:** Evaluate multiple criteria and plan ahead
- **Learning:** Test if sophisticated planning pays off

---

## Implementation Notes

### Base Strategy Class
All strategies inherit from:
```python
class Strategy(ABC):
    def __init__(self, name: str)

    @abstractmethod
    def select_work(self, player, game) -> Optional[Card]

    def get_commissionable_works(self, player) -> List[Card]
```

### Common Patterns
- All strategies first call `get_commissionable_works()` to filter playable cards
- Most strategies have fallback logic (e.g., random choice if primary logic fails)
- Strategies only decide which card to play, not how many (game rules determine that)

### Limitations
Current strategies do NOT consider:
- Future turns or opponent actions
- Optimal VP pool management (when to discard)
- Card draw probabilities
- Opponent board states or competition

These limitations create opportunities for even more sophisticated strategies in future iterations.

---

## Future Strategy Ideas

Potential enhancements for more sophisticated play:

1. **Hybrid Strategy:** Weights multiple objectives based on game state
2. **Adaptive Strategy:** Changes approach based on what's achievable
3. **Opponent-Aware Strategy:** Considers what others are doing
4. **VP Efficiency Strategy:** Optimizes VP spent vs. VP gained ratio
5. **Endgame Strategy:** Different behavior in early vs. late seasons

---

## Usage in Simulations

All strategies are registered in `ALL_STRATEGIES` list and randomly assigned to players via `get_random_strategy()`. This ensures unbiased testing across many games.

To analyze strategy performance, run:
```bash
python examples/run_simulations.py 100000
```

Then examine the results in `examples/modernisme_logs/simulation_summary.txt` to see which strategies consistently outperform others.
