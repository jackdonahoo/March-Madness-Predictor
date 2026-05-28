# March Madness Predictor — ECE 3308

A four-phase machine-learning pipeline that predicts NCAA Tournament outcomes using Dean Oliver's **Four Factors** of basketball, logistic regression, and a 50,000-run Monte Carlo simulation.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Methodology](#methodology)
   - [Phase 1 — Feature Engineering](#phase-1--feature-engineering)
   - [Phase 2 — Exploratory Data Analysis](#phase-2--exploratory-data-analysis)
   - [Phase 3 — Logistic Regression Model](#phase-3--logistic-regression-model)
   - [Phase 4 — Monte Carlo Simulation](#phase-4--monte-carlo-simulation)
4. [Model Performance](#model-performance)
5. [2026 Tournament Predictions](#2026-tournament-predictions)
   - [Predicted Bracket](#predicted-bracket)
   - [Championship Probabilities](#championship-probabilities)
   - [Cinderella Picks](#cinderella-picks)
   - [Portfolio / Value Analysis](#portfolio--value-analysis)
6. [Project Files](#project-files)

---

## Project Overview

This project applies machine learning to bracket prediction across the 2003–2026 NCAA Men's Basketball Tournament. The pipeline is broken into four deliverables, each building on the last:

| Phase | Notebook / Script | Goal |
|-------|-------------------|------|
| 1 | `Master_Analytical_Table.csv` (built externally) | Build per-game Four-Factor differentials |
| 2 | `EDA_Analysis.ipynb` / `Project2.ipynb` | Visualize which features drive wins |
| 3 | `project03.ipynb` | Train and validate a logistic regression classifier |
| 4 | `SimulationScript.py` / `monte_carlo_gauntlet.py` | Simulate the 2026 bracket 50,000 times |

---

## Data Sources

| File | Description |
|------|-------------|
| `Master_Analytical_Table.csv` | Per-game Four-Factor differentials + win label (2003–2025) |
| `march-machine-learning-mania-2026/` | Kaggle NCAA dataset — seeds, slots, detailed results |
| `2026_regular_season_raw.csv` | 2026 regular-season box scores for feature computation |
| `kenpom_2026.csv` | KenPom Adjusted Efficiency Margin for 2026 tournament teams |
| `predictions.csv` | Model predictions on the 2023 test set |
| `frequency_table.csv` | Per-team round probabilities from Monte Carlo simulation |
| `cinderella_report.csv` | Low-seed teams with ≥ 20% Sweet 16 probability |
| `portfolio_analysis.csv` | Value-over-expected analysis for bracket picks |

---

## Methodology

### Phase 1 — Feature Engineering

Four per-game differentials are computed for every matchup (Team A minus Team B):

| Feature | Formula | Meaning |
|---------|---------|---------|
| `DIFF_EFG` | `(FGM + 0.5·FGM3) / FGA` | Shooting efficiency (eFG%) |
| `Diff_TOV` | `TO / (FGA − ORB + TO + 0.475·FTA) × 100` | Turnover rate (%) |
| `DIFF_ORB` | `ORB / (ORB + opp_DRB)` | Offensive rebounding rate |
| `DIFF_FTR` | `FTA / FGA` | Free-throw rate |

These follow Dean Oliver's canonical **Four Factors** framework. A positive differential means Team A has the edge in that category.

---

### Phase 2 — Exploratory Data Analysis

**Figures produced:** `fig1_correlation_matrix.png`, `fig2_boxplots.png`, `fig3_seed_heatmap.png`, `fig4_upset_autopsy.png`

#### Part A — Pearson Correlation Matrix
Measures linear correlation between each Four-Factor differential and `Win_A` across all tournament games (2003–2025). `DIFF_EFG` has the strongest positive correlation — shooting efficiency is the single best predictor of tournament outcomes.

#### Part B — Separation Box Plots
Distributes `DIFF_EFG` and `Diff_TOV` by win/loss outcome. Winners sit clearly above zero on shooting differential; the turnover panel confirms that more turnovers relative to the opponent (positive `Diff_TOV`) hurts win probability.

#### Part C — Seed Geography Heatmap (16 × 16)
Counts every Winner Seed × Loser Seed pairing from 2003–2025. The dense upper-left cluster confirms that high seeds (1–4) beat low seeds most often, while isolated off-diagonal cells mark historic upsets.

#### Part D — Upset Autopsy
Identifies the single largest upset by seed gap, then compares its Four-Factor profile against the average tournament winner. A second panel breaks out upset rate by round — Round 1 upsets are more common than Round 2+.

---

### Phase 3 — Logistic Regression Model

**Notebook:** `project03.ipynb` | **Output:** `predictions.csv`

#### Train / Test Split
- **Train:** all tournament games from seasons 2003–2022
- **Test:** 2023 season only (chronological split prevents data leakage)

#### Features
`DIFF_EFG`, `Diff_TOV`, `DIFF_ORB`, `DIFF_FTR` — all Z-score scaled before fitting.

#### Model Equation

$$\log\text{-odds}(\text{Win}_A) = \beta_0 + \beta_1 \cdot \text{DIFF\_EFG} + \beta_2 \cdot \text{Diff\_TOV} + \beta_3 \cdot \text{DIFF\_ORB} + \beta_4 \cdot \text{DIFF\_FTR}$$

$$P(\text{Win}_A) = \frac{1}{1 + e^{-\log\text{-odds}}}$$

#### Coefficient Interpretation
Since all features are Z-score scaled, coefficient magnitudes are directly comparable:

| Feature | Sign | Meaning |
|---------|------|---------|
| `DIFF_EFG` | **+** (largest) | Shooting advantage is the dominant predictor |
| `Diff_TOV` | **−** | More turnovers than opponent reduces win odds |
| `DIFF_ORB` | **+** | Offensive rebounding edge improves odds |
| `DIFF_FTR` | **+/−** | Weakest contributor; getting to the line matters less in the tournament |

---

### Phase 4 — Monte Carlo Simulation

**Script:** `SimulationScript.py` | **Runs:** 50,000 | **Season:** 2026

#### Architecture

| Part | Description |
|------|-------------|
| A — Bracket Architecture | Loads slot structure from `MNCAATourneySlots.csv`; resolves First Four results |
| B — Simulation Engine | Vectorized NumPy simulation; logistic regression win probability per matchup |
| C — Aggregation | Per-team probabilities for R32, Sweet 16, Elite 8, Final Four, Champion |
| D — Optimal Bracket | Maximum-likelihood winner selected for each slot → `final_bracket.pdf` |

#### Enhanced Features (v2 — SimulationScript)
In addition to the four Dean Oliver factors, the 2026 model adds:
- `DIFF_SEED` — seed differential (A − B)
- `DIFF_MASSEY` — consensus Massey ranking differential
- `DIFF_KENPOM` — KenPom Adjusted Efficiency Margin differential

#### First Four Result Locked In
| Slot | Winner | Loser |
|------|--------|-------|
| Y11 | **Miami (OH)** | Southern Methodist |

---

## Model Performance

Evaluated on the **2023 holdout tournament** (never seen during training):

| Metric | Value | Benchmark | Status |
|--------|-------|-----------|--------|
| Accuracy | see `predictions.csv` | 50% (coin flip) | ✓ PASS |
| Log-Loss | see `predictions.csv` | 0.693 (coin flip) | ✓ < 0.60 target |
| AUC | see `fig3_roc_curve.png` | 0.500 (random) | ✓ > 0.70 target |

**Confusion Matrix:** `fig1_confusion_matrix.png`  
**ROC Curve:** `fig3_roc_curve.png`  
**Coefficient Chart:** `fig2_coefficients.png`

---

## 2026 Tournament Predictions

### Predicted Bracket

The bracket below shows the maximum-likelihood winner for every game, derived from 50,000 simulations.

```
══════════════════════════════════════════════════════════════════════════
                     2026 NCAA TOURNAMENT — PREDICTED BRACKET
══════════════════════════════════════════════════════════════════════════

REGION W                         REGION X
─────────────────────────────    ─────────────────────────────
(1) Duke           ──┐           (1) Florida        ──┐
(16) Siena           ├─ Duke ──┐ (16) Lehigh*         ├─ Florida ──┐
(8) Ohio State     ──┤         │ (8) Clemson         ──┤           │
(9) Texas Christian  ├─ Duke ──┤ (9) Iowa              ├─ Clemson ─┤
(5) St. John's     ──┤         │ (5) Vanderbilt      ──┤           │
(12) Northern Iowa   ├─ Duke ──┤ (12) McNeese State    ├─ Vanderbilt┤
(4) Kansas         ──┤         │ (4) Nebraska        ──┤           │
(13) Cal Baptist     ├─ Kansas ┘ (13) Troy St.          ├─ Nebraska ┘
(6) Louisville     ──┤           (6) North Carolina  ──┤
(11) South Florida   ├─ Conn. ─┐ (11) Virginia Comm.    ├─ Illinois ─┐
(3) Michigan State ──┤         │ (3) Illinois        ──┤            │
(14) N. Dakota St.   ├─ Mich St┤ (14) Pennsylvania     ├─ Illinois ─┤
(7) UCLA           ──┤         │ (7) Saint Mary's CA ──┤            │
(10) UCF             ├─ UCLA  ─┤ (10) Texas A&M         ├─ St Mary's┤
(2) Connecticut    ──┤         │ (2) Houston         ──┤            │
(15) Furman          ├─ Conn. ─┘ (15) Idaho             ├─ Houston  ┘
                     │                                  │
                  DUKE ◄──────────────────────── ILLINOIS
                  (W Region Champion)            (X Region Champion)
                  Champ% 37.89%                  Champ% 5.76%

REGION Y                         REGION Z
─────────────────────────────    ─────────────────────────────
(1) Michigan       ──┐           (1) Arizona         ──┐
(16) Howard*         ├─ Mich. ──┐(16) Long Island U.   ├─ Arizona ──┐
(8) Georgia        ──┤          │(8) Villanova       ──┤            │
(9) Saint Louis      ├─ Georgia ┤(9) Utah State        ├─ Villanova ┤
(5) Texas Tech     ──┤          │(5) Wisconsin       ──┤            │
(12) Akron           ├─ Tx Tech ┤(12) High Point        ├─ Wisconsin ┤
(4) Alabama        ──┤          │(4) Arkansas        ──┤            │
(13) Hofstra         ├─ Alabama ┘(13) Hawai'i           ├─ Arkansas  ┘
(6) Tennessee      ──┤           (6) Brigham Young   ──┤
(11) Miami (OH) ✓    ├─ Tenn. ──┐(11) NC State*         ├─ Purdue ───┐
(3) Virginia       ──┤          │(3) Gonzaga         ──┤             │
(14) Wright State    ├─ Virginia ┤(14) Kennesaw State    ├─ Gonzaga  ─┤
(7) Kentucky       ──┤          │(7) Miami (FL)      ──┤             │
(10) Santa Clara     ├─ S.Clara ┤(10) Missouri          ├─ Miami FL  ┤
(2) Iowa State     ──┤          │(2) Purdue          ──┤             │
(15) Tenn. State     ├─ Iowa St.┘(15) Queens (NC)        ├─ Purdue    ┘
                     │                                   │
               MICHIGAN ◄──────────────────────── ARIZONA
               (Y Region Champion)                (Z Region Champion)
               Champ% 12.87%                       Champ% 12.10%

══════════════════════════════════════════════════════════════════════════
                              FINAL FOUR
══════════════════════════════════════════════════════════════════════════

           DUKE  ──────────────────────────┐
           (W)   Champ% 37.89%             │
                                      ► DUKE
           ILLINOIS ───────────────────────┘
           (X)   Champ% 5.76%

           MICHIGAN ────────────────────────┐
           (Y)   Champ% 12.87%              │
                                       ► DUKE ◄── 🏆 CHAMPION
           ARIZONA ─────────────────────────┘
           (Z)   Champ% 12.10%

══════════════════════════════════════════════════════════════════════════
              🏆  PREDICTED CHAMPION:  DUKE  (37.89%)
══════════════════════════════════════════════════════════════════════════

* Play-In: Howard beats UMBC (Y16) | Lehigh beats Prairie View A&M (X16)
  Miami (OH) beat SMU — CONFIRMED First Four result
  NC State vs Texas (Z11 Play-In — model uses NC State)
```

---

### Championship Probabilities

Full frequency table from 50,000 simulations:

| Rank | Team | Seed | R32 | Sweet 16 | Elite 8 | Final Four | **Champ** |
|------|------|------|-----|----------|---------|------------|-----------|
| 1 | **Duke** | 1 | 99.2% | 91.3% | 83.2% | 72.1% | **37.89%** |
| 2 | Michigan | 1 | 99.1% | 82.7% | 63.8% | 43.6% | 12.87% |
| 3 | Arizona | 1 | 99.2% | 84.9% | 60.5% | 38.2% | 12.10% |
| 4 | Purdue | 2 | 96.0% | 81.8% | 58.4% | 31.9% | 8.55% |
| 5 | Iowa State | 2 | 96.1% | 81.6% | 62.7% | 32.5% | 6.72% |
| 6 | Illinois | 3 | 92.3% | 78.0% | 52.9% | 31.8% | 5.76% |
| 7 | Florida | 1 | 99.3% | 75.8% | 50.2% | 28.2% | 4.30% |
| 8 | Arkansas | 4 | 87.9% | 66.0% | 27.4% | 12.3% | 2.01% |
| 9 | Vanderbilt | 5 | 77.4% | 55.7% | 28.1% | 13.9% | 1.60% |
| 10 | Houston | 2 | 95.8% | 69.1% | 32.0% | 15.4% | 1.60% |
| 11 | Gonzaga | 3 | 91.0% | 65.1% | 27.2% | 11.1% | 1.53% |
| 12 | Connecticut | 2 | 95.7% | 70.5% | 40.1% | 9.4% | 1.16% |
| 13 | Alabama | 4 | 86.2% | 57.7% | 20.0% | 9.3% | 0.96% |
| 14 | Michigan State | 3 | 90.1% | 52.6% | 27.4% | 6.2% | 0.66% |
| 15 | Louisville | 6 | 70.9% | 38.8% | 20.6% | 4.3% | 0.41% |
| 16 | Virginia | 3 | 90.4% | 56.5% | 18.6% | 5.3% | 0.32% |
| 17 | Nebraska | 4 | 86.3% | 37.1% | 11.0% | 3.4% | 0.12% |
| 18 | Iowa | 9 | 58.9% | 17.3% | 7.8% | 2.8% | 0.13% |

> Full table (all 68 teams) in [`frequency_table.csv`](frequency_table.csv)

---

### Cinderella Picks

Low-seed teams (seed ≥ 10) with > 20% Sweet 16 probability, generated by the simulation:

| Team | Seed | R32 | Sweet 16 | Elite 8 | Final Four | Champ | Round 1 Opponent | Edge |
|------|------|-----|----------|---------|------------|-------|-----------------|------|
| **Miami (OH)** | 11 | 40.6% | 16.9% | 4.2% | 1.1% | 0.05% | (6) Tennessee | eFG: 57.6% vs 52.4% — shooting edge |
| **Virginia Commonwealth** | 11 | 45.2% | 10.2% | 3.7% | 1.0% | 0.02% | (6) North Carolina | FTR 0.443 vs 0.380 — gets to the line more |
| **Texas A&M** | 10 | 38.3% | 10.1% | 2.0% | 0.5% | 0.0% | (7) Saint Mary's CA | Composite four-factor edge |
| **Akron** | 12 | 31.0% | 7.8% | 1.1% | 0.2% | 0.0% | (5) Texas Tech | ORB% 52.7% — offensive glass edge |
| **South Florida** | 11 | 29.2% | 7.7% | 1.9% | 0.2% | 0.0% | (6) Louisville | FTR 0.424 vs 0.341 — foul-drawing edge |

> Full report in [`cinderella_report.csv`](cinderella_report.csv)

---

### Portfolio / Value Analysis

Compares simulation championship probability against seed-based expected probability to find over/under-valued picks:

**Value Ratio = Sim% / Expected%** — values > 1.0 mean the model likes a team more than their seed implies.

| Team | Seed | Sim Champ% | Expected% | **Champ Value Ratio** | F4 Value Ratio |
|------|------|------------|-----------|----------------------|----------------|
| **Duke** | 1 | 37.89% | 16.0% | **2.37** | 1.80 |
| **Illinois** | 3 | 5.76% | 3.8% | **1.52** | 2.27 |
| **Vanderbilt** | 5 | 1.60% | 1.2% | **1.33** | 2.40 |
| **Purdue** | 2 | 8.55% | 6.5% | **1.32** | 1.39 |
| **Iowa State** | 2 | 6.72% | 6.5% | **1.03** | 1.41 |
| Arkansas | 4 | 2.01% | 2.2% | 0.91 | 1.29 |
| Michigan | 1 | 12.87% | 16.0% | 0.80 | 1.09 |
| Arizona | 1 | 12.10% | 16.0% | 0.76 | 0.95 |
| Florida | 1 | 4.30% | 16.0% | 0.27 | 0.70 |
| Houston | 2 | 1.60% | 6.5% | 0.25 | 0.67 |
| Connecticut | 2 | 1.16% | 6.5% | 0.18 | 0.41 |

> Full table in [`portfolio_analysis.csv`](portfolio_analysis.csv)

**Key insight:** Duke is the top value pick at 2.37× expected — the model sees the Blue Devils as significantly stronger than their 1-seed peers. Illinois (3 seed, 1.52×) and Vanderbilt (5 seed, 1.33×) are the best non-top-seed value plays. Florida (1 seed, 0.27×) is the most over-seeded team in the field.

---

## Project Files

### Notebooks & Scripts

| File | Description |
|------|-------------|
| [`EDA_Analysis.ipynb`](EDA_Analysis.ipynb) | Clean EDA notebook — correlation matrix, box plots, heatmap, upset autopsy |
| [`Project2.ipynb`](Project2.ipynb) | Alternate EDA implementation with additional annotation |
| [`project03.ipynb`](project03.ipynb) | Logistic regression training, evaluation, coefficient analysis |
| [`SimulationScript.py`](SimulationScript.py) | 2026 Monte Carlo simulation (50,000 runs, 7 features) |
| [`monte_carlo_gauntlet.py`](monte_carlo_gauntlet.py) | 2025 Monte Carlo prototype (10,000 runs, 4 features) |

### Output Files

| File | Description |
|------|-------------|
| [`final_bracket.pdf`](final_bracket.pdf) | Visual 63-game predicted bracket (generated by simulation) |
| [`alpha_report.pdf`](alpha_report.pdf) | Full analysis report |
| [`predictions.csv`](predictions.csv) | Model predictions on 2023 test set with probabilities |
| [`frequency_table.csv`](frequency_table.csv) | Per-team round-by-round probabilities (all 68 teams) |
| [`cinderella_report.csv`](cinderella_report.csv) | Underdog teams with meaningful Sweet 16 chances |
| [`portfolio_analysis.csv`](portfolio_analysis.csv) | Value-over-expected analysis for bracket strategy |
| [`Master_Analytical_Table.csv`](Master_Analytical_Table.csv) | Full training dataset (2003–2025, all matchups) |
| [`kenpom_2026.csv`](kenpom_2026.csv) | KenPom Adjusted Efficiency Margin ratings for 2026 |
| [`2026_regular_season_raw.csv`](2026_regular_season_raw.csv) | Raw 2026 box scores for feature computation |

### Figures

| Figure | Description |
|--------|-------------|
| [`fig1_correlation_matrix.png`](fig1_correlation_matrix.png) | Pearson correlation — Four Factors vs Win_A |
| [`fig2_boxplots.png`](fig2_boxplots.png) | eFG% and TOV% distributions by win/loss |
| [`fig3_seed_heatmap.png`](fig3_seed_heatmap.png) | 16×16 winner–loser seed geography heatmap |
| [`fig4_upset_autopsy.png`](fig4_upset_autopsy.png) | Biggest upset Four-Factor breakdown + upset rate by round |
| [`fig1_confusion_matrix.png`](fig1_confusion_matrix.png) | Logistic regression confusion matrix (2023 test) |
| [`fig2_coefficients.png`](fig2_coefficients.png) | Logistic regression coefficient bar chart |
| [`fig3_roc_curve.png`](fig3_roc_curve.png) | ROC curve with AUC score |
| [`proof_of_life.png`](proof_of_life.png) | Simulation proof-of-life output |

### Data (Kaggle NCAA Dataset)

Located in [`march-machine-learning-mania-2026/`](march-machine-learning-mania-2026/):

- `MRegularSeasonDetailedResults.csv` — historical box scores
- `MNCAATourneyDetailedResults.csv` — tournament box scores
- `MNCAATourneySeeds.csv` — seed assignments by team and year
- `MNCAATourneySlots.csv` — bracket slot structure (defines matchup order)
- `MNCAATourneySeedRoundSlots.csv` — seed × round slot table
- `MRegularSeasonCompactResults.csv` / `MNCAATourneyCompactResults.csv` — compact results
- `MSeasons.csv`, `MTeamConferences.csv`, `MGameCities.csv` — metadata
- `SampleSubmissionStage1.csv` / `SampleSubmissionStage2.csv` — Kaggle submission formats

---

## Quick Start

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn reportlab

# Run the 2026 simulation
python SimulationScript.py

# Outputs:
#   frequency_table.csv    — round probabilities for all teams
#   cinderella_report.csv  — underdog analysis
#   portfolio_analysis.csv — value picks
#   final_bracket.pdf      — visual bracket
```

---

*ECE 3308 — Probabilistic Methods in Engineering*
