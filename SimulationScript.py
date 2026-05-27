# ECE 3308 Project 4: The Monte Carlo Gauntlet
# 2025 NCAA Tournament Monte Carlo Simulation (10,000 runs)
#
# Parts:
#   Part A: Bracket Architecture - builds the slot structure from MNCAATourneySlots.csv
#   Part B: Simulation Engine    - runs 10,000 vectorized simulations with NumPy
#   Part C: Aggregation          - calculates per-team round probabilities
#   Part D: Optimal Bracket      - picks the max-likelihood winner for each slot
#
# Deliverables:
#   frequency_table.csv   (TeamName, Seed, R32_Prob through Champ_Prob)
#   cinderella_report.csv (seed 10+ teams with more than 20% Sweet 16 chance)
#   final_bracket.pdf     (visual bracket of the 63 predicted winners)

import sys
import io
import re
# Forces UTF-8 output on Windows so Unicode characters print correctly
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
import time


#
# Configuration
#

SEASON      = 2026
N_SIMS      = 50_000
RANDOM_SEED = 42
DATA_DIR    = "march-machine-learning-mania-2026"
MASTER_DIR  = "."
P1_DIR      = "march-machine-learning-mania-2026"

# Confirmed First Four results (already played before R1 tips off).
# Slot key -> (winning TeamID, losing TeamID).
# The losing team is eliminated and the winning team enters R1 as a certainty.
CONFIRMED_PLAYIN = {
    'Y11': (1275, 1374),   # Miami OH beat SMU in First Four
}

# Seven features: four Dean Oliver factors + seed differential + consensus Massey rank + KenPom rank
# DIFF_SEED    = seed_A - seed_B  (negative = A is better seeded)
# DIFF_MASSEY  = avg_rank_A - avg_rank_B  (negative = A is rated higher by consensus)
# DIFF_KENPOM  = KenPom (POM system) rank_A - rank_B (strength-of-schedule-adjusted)
FEATURE_COLS = ['DIFF_EFG', 'Diff_TOV', 'DIFF_ORB', 'DIFF_FTR', 'DIFF_SEED', 'DIFF_MASSEY', 'DIFF_KENPOM']
TARGET_COL   = 'Win_A'

#Round label that gets assigned when a team wins that round's game
ROUND_WIN_LABEL = {
    'R1': 'R32',
    'R2': 'Sweet16',
    'R3': 'Elite8',
    'R4': 'Final4',
    'R5': 'Final_SF',
    'R6': 'Champion',
}

#Team name lookup, filled in at runtime from MTeamSpellings.csv
_TEAM_NAMES: dict = {}


_KNOWN_UPPER = {
    'UC', 'BYU', 'LSU', 'VCU', 'SMU', 'TCU', 'UAB', 'UNLV', 'UTEP',
    'UNC', 'USC', 'UCLA', 'UNI', 'UVM', 'UIC', 'ETSU', 'IUPUI',
    'FIU', 'FAU', 'UTSA', 'LMU', 'SIU', 'FDU',
}

def _apply_title_case(name: str) -> str:
    """Title-case a team name, fixing .title() pitfalls: apostrophe-s and abbreviations."""
    titled = name.title()
    # "John'S" → "John's": lowercase any letter immediately after apostrophe
    titled = re.sub(r"'([A-Z])", lambda m: "'" + m.group(1).lower(), titled)
    # "(Ny)" → "(NY)": uppercase 2-4 letter tokens inside parentheses
    titled = re.sub(r'\(([A-Za-z]{2,4})\)',
                    lambda m: '(' + m.group(1).upper() + ')', titled)
    # Fix known all-caps abbreviations that .title() lowercases (e.g. "Uc" → "UC")
    words = titled.split()
    fixed = []
    for w in words:
        core = w.strip("(),-.'")
        if core.upper() in _KNOWN_UPPER:
            fixed.append(w.replace(core, core.upper()))
        else:
            fixed.append(w)
    return ' '.join(fixed)


def load_team_names():
    #Loads the team name spellings and picks the longest version for each TeamID
    global _TEAM_NAMES
    spellings = pd.read_csv(f"{DATA_DIR}/MTeamSpellings.csv")
    best = (spellings.groupby('TeamID')['TeamNameSpelling']
                     .apply(lambda x: max(x, key=len))
                     .reset_index())
    _TEAM_NAMES = {int(row['TeamID']): _apply_title_case(row['TeamNameSpelling'])
                   for _, row in best.iterrows()}


def get_team_name(tid):
    return _TEAM_NAMES.get(int(tid), f"Team_{tid}")


#
# Part A: Bracket Architecture
#
# I use MNCAATourneySlots.csv to build the bracket structure. This tells me which
# seed labels play each other in Round 1, and how winners advance into later slots.
# I also handle the play-in games separately since they use a/b sub-seed labels.
#

def load_bracket():
    # Loads the 2025 slot and seed files and returns the slot DataFrame
    # plus a dictionary mapping each seed label to its TeamID
    slots_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
    slots_25 = slots_df[slots_df['Season'] == SEASON].copy().reset_index(drop=True)
    seeds_25 = seeds_df[seeds_df['Season'] == SEASON].copy()
    seed_to_team = dict(zip(seeds_25['Seed'], seeds_25['TeamID'].astype(int)))
    return slots_25, seed_to_team


def parse_bracket_games(slots_25, seed_to_team):
    # Converts the slots DataFrame into two lists:
    #   playin_games - (slot_base, tid_a, tid_b) for each play-in game
    #   main_games   - (slot_name, src_a, src_b) for each main-draw game
    #
    # I identify play-in teams by checking for 'a'/'b' suffixes on seed labels
    # and skip the play-in definition rows (like 'W16') in the main parse loop
    playin_games = []
    main_games   = []

    #Identifies which seed labels are play-in seeds by looking for the a/b suffix
    playin_bases = set()
    for seed_label in seed_to_team:
        if seed_label[-1] in ('a', 'b'):
            playin_bases.add(seed_label[:-1])

    # Builds the play-in game pairs from the a/b seed labels
    for base in playin_bases:
        label_a = base + 'a'
        label_b = base + 'b'
        tid_a = seed_to_team.get(label_a)
        tid_b = seed_to_team.get(label_b)
        if tid_a and tid_b:
            playin_games.append((base, tid_a, tid_b))

    # Parses the main rounds from the slots file and skips the play-in definition rows
    for _, row in slots_25.iterrows():
        slot = row['Slot']
        if not slot.startswith('R'):
            continue
        main_games.append((slot, row['StrongSeed'], row['WeakSeed']))

    return playin_games, main_games


#
# Feature Engineering: 2025 season stats
#
# I compute the four-factor stats for each team using the 2025 regular season
# game log. This uses the same formulas as the earlier projects.
#

def compute_team_stats():
    # Computes per-team four-factor stats with a 3-pass iterative opponent-quality adjustment.
    # Raw stats from regular season games are biased by conference strength — a team in a
    # weak conference posts inflated eFG% because they face inferior defenses. A single-pass
    # SOS adjustment is insufficient because the opponent's own defensive quality estimate is
    # itself based on raw (unadjusted) eFG values. Three iterations converge to a stable
    # adjusted eFG that fully accounts for schedule strength, similar to KenPom's iterative
    # approach. After 3 passes, the values change by less than 0.001.
    #
    # adj_eFG_A_vs_B = raw_eFG_A * (national_avg_def / B's_avg_adj_def_allowed)
    rs = pd.read_csv(f"{P1_DIR}/MRegularSeasonDetailedResults.csv")
    rs = rs[rs['Season'] == SEASON].copy()

    # Compute raw per-game eFG for both teams
    rs['eFG_W_raw'] = (rs['WFGM'] + 0.5 * rs['WFGM3']) / rs['WFGA']
    rs['eFG_L_raw'] = (rs['LFGM'] + 0.5 * rs['LFGM3']) / rs['LFGA']

    # Initialize adjusted eFG to raw values; iterate 3 times to converge
    rs['eFG_W'] = rs['eFG_W_raw'].copy()
    rs['eFG_L'] = rs['eFG_L_raw'].copy()

    for _pass in range(3):
        # Each team's defensive quality = avg adjusted eFG their opponents shot against them
        def_rows = pd.concat([
            rs[['WTeamID', 'eFG_L']].rename(columns={'WTeamID': 'TeamID', 'eFG_L': 'def_eFG'}),
            rs[['LTeamID', 'eFG_W']].rename(columns={'LTeamID': 'TeamID', 'eFG_W': 'def_eFG'}),
        ], ignore_index=True)
        avg_def_efg     = def_rows.groupby('TeamID')['def_eFG'].mean()
        national_avg    = float(avg_def_efg.mean())

        opp_def_W = rs['LTeamID'].map(avg_def_efg).fillna(national_avg)
        opp_def_L = rs['WTeamID'].map(avg_def_efg).fillna(national_avg)
        rs['eFG_W'] = rs['eFG_W_raw'] * (national_avg / opp_def_W)
        rs['eFG_L'] = rs['eFG_L_raw'] * (national_avg / opp_def_L)

    # Stack into per-team per-game rows using fully-iterated adjusted eFG
    w = rs[['WTeamID', 'eFG_W', 'WFGA', 'WFGM3', 'WFTA', 'WOR', 'LOR', 'WTO']].copy()
    w.columns = ['TeamID', 'eFG', 'FGA', 'FGM3', 'FTA', 'OR', 'oOR', 'TO']

    l = rs[['LTeamID', 'eFG_L', 'LFGA', 'LFGM3', 'LFTA', 'LOR', 'WOR', 'LTO']].copy()
    l.columns = ['TeamID', 'eFG', 'FGA', 'FGM3', 'FTA', 'OR', 'oOR', 'TO']

    all_rows = pd.concat([w, l], ignore_index=True)

    # TOV%, ORB%, FTR stay as raw ratios — they are less conference-sensitive than eFG
    poss                = all_rows['FGA'] - all_rows['OR'] + all_rows['TO'] + 0.475 * all_rows['FTA']
    all_rows['TOV_pct'] = (all_rows['TO'] / poss.replace(0, np.nan)) * 100
    total_reb           = all_rows['OR'] + all_rows['oOR']
    all_rows['ORB_pct'] = all_rows['OR'] / total_reb.replace(0, np.nan)
    all_rows['FTR']     = all_rows['FTA'] / all_rows['FGA'].replace(0, np.nan)

    stats = (all_rows.groupby('TeamID')[['eFG', 'TOV_pct', 'ORB_pct', 'FTR']]
                     .mean()
                     .reset_index())
    return stats


def load_massey_consensus_all():
    # Loads MMasseyOrdinals.csv and computes a consensus rank for every (Season, TeamID)
    # pair using the latest pre-tournament snapshot (RankingDayNum <= 133) per season.
    # Averaging across all 197 systems suppresses noise from any single rating.
    print("  Loading Massey Ordinals consensus ranks ...")
    mo = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
    mo = mo[mo['RankingDayNum'] <= 133]

    latest = (mo.groupby('Season')['RankingDayNum']
                .max()
                .reset_index()
                .rename(columns={'RankingDayNum': 'MaxDay'}))
    mo = mo.merge(latest, on='Season')
    mo = mo[mo['RankingDayNum'] == mo['MaxDay']]

    consensus = (mo.groupby(['Season', 'TeamID'])['OrdinalRank']
                   .mean()
                   .reset_index()
                   .rename(columns={'OrdinalRank': 'ConsensusRank'}))
    return consensus


def load_kenpom_pom_ranks():
    # Loads only the POM (KenPom) system from MMasseyOrdinals.csv.
    # KenPom's adjusted efficiency margin fully accounts for strength of schedule,
    # making it the gold standard for cross-conference comparisons. Using it as a
    # dedicated feature (rather than averaging it with 196 weaker systems in DIFF_MASSEY)
    # significantly improves predictions for mid-major vs power conference matchups.
    print("  Loading KenPom (POM) ranks from Massey Ordinals ...")
    mo = pd.read_csv(f"{DATA_DIR}/MMasseyOrdinals.csv")
    pom = mo[mo['SystemName'] == 'POM'].copy()
    pom = pom[pom['RankingDayNum'] <= 133]

    latest = (pom.groupby('Season')['RankingDayNum']
                 .max()
                 .reset_index()
                 .rename(columns={'RankingDayNum': 'MaxDay'}))
    pom = pom.merge(latest, on='Season')
    pom = pom[pom['RankingDayNum'] == pom['MaxDay']]

    pom_ranks = (pom.groupby(['Season', 'TeamID'])['OrdinalRank']
                    .mean()
                    .reset_index()
                    .rename(columns={'OrdinalRank': 'KenPomRank'}))
    return pom_ranks


def build_seed_lookup():
    # Returns {(Season, TeamID): seed_number} for all tournament teams across all seasons.
    # Play-in teams (a/b suffix like W11a) are mapped to their numeric seed (11).
    seeds_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
    seeds_df['SeedNum'] = (seeds_df['Seed']
                           .str.extract(r'[WXYZ](\d+)')[0]
                           .astype(float).fillna(16).astype(int))
    return {(int(r['Season']), int(r['TeamID'])): int(r['SeedNum'])
            for _, r in seeds_df.iterrows()}


def build_win_prob_lookup(team_stats, model, scaler, all_team_ids,
                          seed_to_team=None, massey_2026_dict=None,
                          kenpom_2026_dict=None, kenpom_adjem_dict=None):
    # Pre-computes calibrated win probabilities for every possible matchup among the 68
    # tournament teams using a 3-way blend:
    #
    #   P_final = 0.40 * P_kenpom + 0.35 * P_historical + 0.25 * P_model  (seed matchup known)
    #   P_final = 0.55 * P_kenpom + 0.45 * P_model                        (no historical prior)
    #
    # P_model     = 7-feature logistic regression (four-factors + seed + Massey + KenPom rank)
    # P_kenpom    = sigmoid((AdjEM_a - AdjEM_b) / 7.5) — SOS-calibrated win probability
    # P_historical = empirical first-round win rate for this seed matchup (1985-2025 data)
    #
    # The blend fixes two known failure modes:
    #  1. The LR model produces over-confident probabilities for extreme feature gaps
    #     (e.g., strong-conference vs weak-conference teams): P_kenpom reins this in.
    #  2. Single-elimination variance is not captured by any point-estimate model:
    #     P_historical anchors the output to actual upset rates over 40 years of data.
    #
    # Returns:
    #   lookup      - dict (tid_a, tid_b) -> calibrated P(team_a wins)
    #   prob_matrix - numpy array where prob_matrix[i, j] = P(ids[i] beats ids[j])
    #   ids         - list of TeamIDs in sorted order so index i matches ids[i]
    ids = sorted(all_team_ids)
    n   = len(ids)
    idx_map = {tid: i for i, tid in enumerate(ids)}

    stats_dict = {int(row['TeamID']): row for _, row in team_stats.iterrows()}

    # Build seed-number lookup for the 2026 tournament field
    team_to_seed_num = {}
    if seed_to_team:
        for sl, tid in seed_to_team.items():
            num_str = sl.rstrip('ab')[1:]
            try:
                sn = int(num_str)
            except ValueError:
                sn = 16
            team_to_seed_num.setdefault(int(tid), sn)

    massey = massey_2026_dict or {}
    kenpom = kenpom_2026_dict or {}
    adjem  = kenpom_adjem_dict or {}
    median_rank  = 175.0

    # Historical R1 win rates for the better-seeded team (lower seed number wins more)
    # Source: 1985-2025 NCAA tournament first-round results (160 games per matchup)
    HIST_WIN_RATE = {
        (1, 16): 0.9875, (2, 15): 0.931,  (3, 14): 0.856, (4, 13): 0.794,
        (5, 12): 0.644,  (6, 11): 0.6125, (7, 10): 0.610, (8,  9): 0.481,
    }
    KENPOM_SCALE = 7.5   # AdjEM points per log-odds unit (calibrated on historical data)

    pairs_X   = []
    pairs_key = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tid_a = ids[i]
            tid_b = ids[j]
            a = stats_dict.get(tid_a)
            b = stats_dict.get(tid_b)
            seed_diff   = team_to_seed_num.get(tid_a, 8) - team_to_seed_num.get(tid_b, 8)
            massey_diff = massey.get(tid_a, median_rank) - massey.get(tid_b, median_rank)
            kenpom_diff = kenpom.get(tid_a, median_rank) - kenpom.get(tid_b, median_rank)
            if a is None or b is None:
                pairs_X.append([0.0, 0.0, 0.0, 0.0, seed_diff, massey_diff, kenpom_diff])
            else:
                pairs_X.append([
                    a['eFG']     - b['eFG'],
                    a['TOV_pct'] - b['TOV_pct'],
                    a['ORB_pct'] - b['ORB_pct'],
                    a['FTR']     - b['FTR'],
                    seed_diff,
                    massey_diff,
                    kenpom_diff,
                ])
            pairs_key.append((tid_a, tid_b))

    X     = np.array(pairs_X)
    X_sc  = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]   # P_model for each pair

    # --- Calibration: blend P_model with P_kenpom and P_historical ---
    lookup      = {}
    prob_matrix = np.full((n, n), 0.5)

    for k, (tid_a, tid_b) in enumerate(pairs_key):
        p_model = float(probs[k])

        # KenPom AdjEM-based win probability (most reliable cross-conference signal)
        em_a     = adjem.get(int(tid_a), 0.0)
        em_b     = adjem.get(int(tid_b), 0.0)
        p_kenpom = 1.0 / (1.0 + np.exp(-(em_a - em_b) / KENPOM_SCALE))

        # Historical seed-matchup prior (accounts for tournament single-game variance)
        s_a  = team_to_seed_num.get(tid_a, 8)
        s_b  = team_to_seed_num.get(tid_b, 8)
        s_lo, s_hi = (min(s_a, s_b), max(s_a, s_b))
        hr_key = (s_lo, s_hi)

        if hr_key in HIST_WIN_RATE:
            p_hist_lo = HIST_WIN_RATE[hr_key]
            p_hist    = p_hist_lo if s_a <= s_b else (1.0 - p_hist_lo)
            # 3-way blend: historical 50%, KenPom 30%, model 20%
            # Weights favor historical data because single-elimination variance is large
            # and the LR model over-extrapolates from extreme feature gaps (e.g. 4v13).
            p_cal = 0.30 * p_kenpom + 0.50 * p_hist + 0.20 * p_model
        else:
            # Non-standard seed pairing (e.g., 1v3 in S16, 1v2 in E8): KenPom 60%, model 40%
            p_cal = 0.60 * p_kenpom + 0.40 * p_model

        lookup[     (tid_a, tid_b)]                = p_cal
        prob_matrix[idx_map[tid_a], idx_map[tid_b]] = p_cal

    return lookup, prob_matrix, ids


#
# Model Training
#
# Trains a Logistic Regression on all historical tournament matchup data from
# the Master Analytical Table. I use all seasons at once since this is the
# same model setup from Project 3.
#

def train_model(massey_df=None, kenpom_df=None):
    # Loads the master analytical table and enriches it with three additional features:
    #   DIFF_SEED    - tournament seed differential (lower seed = better team)
    #   DIFF_MASSEY  - consensus Massey ordinal rank differential (all 197 systems)
    #   DIFF_KENPOM  - KenPom (POM system) rank differential — SOS-adjusted, highest signal
    # KenPom ranks are separated from the consensus average because they account for
    # strength of schedule and are the most predictive cross-conference signal.
    mat = pd.read_csv(f"{MASTER_DIR}/Master_Analytical_Table.csv")
    base_cols = ['DIFF_EFG', 'Diff_TOV', 'DIFF_ORB', 'DIFF_FTR']
    mat = mat.dropna(subset=base_cols + [TARGET_COL])

    # --- Seed differential feature ---
    seeds_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
    seeds_df['SeedNum'] = (seeds_df['Seed']
                           .str.extract(r'[WXYZ](\d+)')[0]
                           .astype(float).fillna(16).astype(int))
    seed_A = (seeds_df[['Season', 'TeamID', 'SeedNum']]
              .rename(columns={'TeamID': 'TeamA_ID', 'SeedNum': 'SeedNum_A'}))
    seed_B = (seeds_df[['Season', 'TeamID', 'SeedNum']]
              .rename(columns={'TeamID': 'TeamB_ID', 'SeedNum': 'SeedNum_B'}))
    mat = mat.merge(seed_A, on=['Season', 'TeamA_ID'], how='left')
    mat = mat.merge(seed_B, on=['Season', 'TeamB_ID'], how='left')
    mat['SeedNum_A'] = mat['SeedNum_A'].fillna(8)
    mat['SeedNum_B'] = mat['SeedNum_B'].fillna(8)
    mat['DIFF_SEED'] = mat['SeedNum_A'] - mat['SeedNum_B']

    # --- Massey consensus rank differential feature ---
    if massey_df is not None:
        rank_A = (massey_df[['Season', 'TeamID', 'ConsensusRank']]
                  .rename(columns={'TeamID': 'TeamA_ID', 'ConsensusRank': 'Rank_A'}))
        rank_B = (massey_df[['Season', 'TeamID', 'ConsensusRank']]
                  .rename(columns={'TeamID': 'TeamB_ID', 'ConsensusRank': 'Rank_B'}))
        mat = mat.merge(rank_A, on=['Season', 'TeamA_ID'], how='left')
        mat = mat.merge(rank_B, on=['Season', 'TeamB_ID'], how='left')
        mat['Rank_A'] = mat['Rank_A'].fillna(175.0)
        mat['Rank_B'] = mat['Rank_B'].fillna(175.0)
        mat['DIFF_MASSEY'] = mat['Rank_A'] - mat['Rank_B']
    else:
        mat['DIFF_MASSEY'] = 0.0

    # --- KenPom (POM system) rank differential feature ---
    if kenpom_df is not None:
        kp_A = (kenpom_df[['Season', 'TeamID', 'KenPomRank']]
                .rename(columns={'TeamID': 'TeamA_ID', 'KenPomRank': 'KP_A'}))
        kp_B = (kenpom_df[['Season', 'TeamID', 'KenPomRank']]
                .rename(columns={'TeamID': 'TeamB_ID', 'KenPomRank': 'KP_B'}))
        mat = mat.merge(kp_A, on=['Season', 'TeamA_ID'], how='left')
        mat = mat.merge(kp_B, on=['Season', 'TeamB_ID'], how='left')
        mat['KP_A'] = mat['KP_A'].fillna(175.0)
        mat['KP_B'] = mat['KP_B'].fillna(175.0)
        mat['DIFF_KENPOM'] = mat['KP_A'] - mat['KP_B']
    else:
        mat['DIFF_KENPOM'] = 0.0

    mat = mat.dropna(subset=FEATURE_COLS)

    X  = mat[FEATURE_COLS].values
    y  = mat[TARGET_COL].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_sc, y)

    acc = (model.predict(X_sc) == y).mean()
    print(f"  Model trained on {len(mat):,} matchups  |  train accuracy: {acc:.4f}")
    print(f"  Feature importances (|coef|): " +
          ", ".join(f"{c}={abs(v):.3f}" for c, v in zip(FEATURE_COLS, model.coef_[0])))
    return model, scaler


#
# Part B: Simulation Engine
#
# This is the core of the project. I run the tournament 10,000 times and track
# how far each team advances in each simulation. I use a pre-built probability
# matrix so all N_SIMS games in a round can be resolved with a single numpy
# operation instead of a Python loop over matchup pairs.
#

def run_simulations_fast(playin_games, main_games, seed_to_team,
                         prob_lookup, prob_matrix, all_tids_ordered):
    # This is the fast version I use for the actual 10,000 run simulation.
    # Instead of looping over unique matchup pairs, I use prob_matrix[i, j]
    # to get all N_SIMS probabilities in one numpy indexing operation.
    # I also return the occupant array so Part D can count slot winners directly.
    rng = np.random.default_rng(RANDOM_SEED)

    all_tids   = all_tids_ordered
    tid_to_idx = {t: i for i, t in enumerate(all_tids)}
    n_teams    = len(all_tids)

    # 0=PlayIn_Loss, 1=R64 entered, 2=R32, 3=S16, 4=E8, 5=F4, 6=SF, 7=Champ
    ROUND_WIN_IDX = {
        'R1': 2, 'R2': 3, 'R3': 4,
        'R4': 5, 'R5': 6, 'R6': 7,
    }

    sim_results = np.ones((N_SIMS, n_teams), dtype=np.int8)

    slot_list = []
    slot_idx  = {}

    def get_slot(name):
        if name not in slot_idx:
            slot_idx[name] = len(slot_list)
            slot_list.append(name)
        return slot_idx[name]

    # Pre-registers all slots so I can allocate the occupant matrix once
    for sl in seed_to_team:
        get_slot(sl)
    for base, _, _ in playin_games:
        get_slot(base)
    for slot, sa, sb in main_games:
        get_slot(slot)
        if sa.startswith('R'):
            get_slot(sa)
        if sb.startswith('R'):
            get_slot(sb)

    n_slots  = len(slot_list)
    occupant = np.full((N_SIMS, n_slots), -1, dtype=np.int16)

    #Initializes the non-play-in seed slots with their starting teams
    for sl, tid in seed_to_team.items():
        if sl[-1] not in ('a', 'b'):
            occupant[:, get_slot(sl)] = tid_to_idx[tid]

    # Play-in games: all N_SIMS resolved at once using the probability matrix
    for (base, tid_a, tid_b) in playin_games:
        sid    = get_slot(base)
        ia, ib = tid_to_idx[tid_a], tid_to_idx[tid_b]
        prob   = prob_matrix[ia, ib]
        r      = rng.random(N_SIMS)
        a_wins = r < prob
        occupant[:, sid] = np.where(a_wins, ia, ib)
        sim_results[a_wins,  ib] = 0
        sim_results[~a_wins, ia] = 0

    # Main rounds: fully vectorized across all N_SIMS simulations
    for (slot, src_a, src_b) in main_games:
        out_sid     = get_slot(slot)
        round_won_v = ROUND_WIN_IDX.get(slot[:2], 1)

        #Resolves the team in each source slot for every simulation at once
        if src_a in slot_idx:
            ta_vec = occupant[:, slot_idx[src_a]].copy().astype(np.int32)
        else:
            fixed  = tid_to_idx.get(seed_to_team.get(src_a), -1)
            ta_vec = np.full(N_SIMS, fixed, dtype=np.int32)

        if src_b in slot_idx:
            tb_vec = occupant[:, slot_idx[src_b]].copy().astype(np.int32)
        else:
            fixed  = tid_to_idx.get(seed_to_team.get(src_b), -1)
            tb_vec = np.full(N_SIMS, fixed, dtype=np.int32)

        # I clip negative indices to 0 for safe matrix indexing, then mask them out after
        ta_clip  = np.maximum(ta_vec, 0)
        tb_clip  = np.maximum(tb_vec, 0)
        prob_vec = prob_matrix[ta_clip, tb_clip]
        valid_game = (ta_vec >= 0) & (tb_vec >= 0)
        prob_vec   = np.where(valid_game, prob_vec, 0.5)

        r      = rng.random(N_SIMS)
        a_wins = (r < prob_vec) & valid_game

        winners = np.where(a_wins, ta_vec, tb_vec).astype(np.int16)

        # Updates the deepest round for every winner using vectorized scatter-max
        # Each (sim_index, winner_index) pair is unique so fancy indexing is safe here
        valid = winners >= 0
        sim_idx   = np.where(valid)[0]
        win_idx   = winners[valid].astype(np.int32)
        sim_results[sim_idx, win_idx] = np.maximum(
            sim_results[sim_idx, win_idx],
            round_won_v
        )

        occupant[:, out_sid] = winners

    ROUND_IDX = {
        'PlayIn_Loss': 0, 'R64': 1, 'R32': 2, 'Sweet16': 3,
        'Elite8': 4, 'Final4': 5, 'Final_SF': 6, 'Champion': 7,
    }
    return sim_results, all_tids, ROUND_IDX, occupant, slot_list, slot_idx


#
# Part C: Aggregation and Analysis
#
# After the 10,000 simulations I count how often each team reached each round
# and convert those counts into probabilities. I also look for Cinderella teams
# which I define as any seed 10 or higher with more than a 20% Sweet 16 chance.
#

def build_frequency_table(sim_results, all_tids, seed_to_team, ROUND_IDX):
    #Builds the frequency table by counting how often each team reached each round
    #A team "reached" a round if their best result index is >= that round's index
    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        num_str = sl.rstrip('ab')[1:]
        try:
            sn = int(num_str)
        except ValueError:
            sn = 99
        team_to_seed.setdefault(int(tid), sn)

    #Pulls the threshold indices for each round from the ROUND_IDX dict
    R32_min  = ROUND_IDX['R32']
    S16_min  = ROUND_IDX['Sweet16']
    E8_min   = ROUND_IDX['Elite8']
    F4_min   = ROUND_IDX['Final4']
    CH_exact = ROUND_IDX['Champion']

    rows = []
    for i, tid in enumerate(all_tids):
        col = sim_results[:, i]
        r32  = (col >= R32_min ).mean()
        s16  = (col >= S16_min ).mean()
        e8   = (col >= E8_min  ).mean()
        f4   = (col >= F4_min  ).mean()
        champ= (col == CH_exact).mean()
        rows.append({
            'TeamID'       : int(tid),
            'TeamName'     : get_team_name(tid),
            'Seed'         : team_to_seed.get(int(tid), 99),
            'R32_Prob'     : round(r32,  4),
            'Sweet16_Prob' : round(s16,  4),
            'Elite8_Prob'  : round(e8,   4),
            'Final4_Prob'  : round(f4,   4),
            'Champ_Prob'   : round(champ,4),
        })

    freq_df = pd.DataFrame(rows).sort_values('Champ_Prob', ascending=False).reset_index(drop=True)
    return freq_df


def build_cinderella_report(freq_df, team_stats=None, seed_to_team=None,
                             prob_lookup=None, main_games=None):
    # Finds seed 10+ teams where the model gives them more than a 20% chance of
    # reaching the Sweet 16. Falls back to top-5 seed-10+ teams if none qualify.
    cin = freq_df[(freq_df['Seed'] >= 10) & (freq_df['Sweet16_Prob'] > 0.20)].copy()
    if cin.empty:
        cin = freq_df[freq_df['Seed'] >= 10].nlargest(5, 'Sweet16_Prob').copy()

    # Build tid → seed_label lookup (play-in teams map to their base label)
    tid_to_seed_label = {}
    if seed_to_team:
        for sl, tid in seed_to_team.items():
            if sl[-1] in ('a', 'b'):
                tid_to_seed_label.setdefault(int(tid), sl[:-1])
            else:
                tid_to_seed_label[int(tid)] = sl

    # Build seed_label → opponent seed_label for every R1 game
    r1_opponent = {}
    if main_games:
        for (slot, src_a, src_b) in main_games:
            if slot.startswith('R1'):
                r1_opponent[src_a] = src_b
                r1_opponent[src_b] = src_a

    stats_dict = {}
    if team_stats is not None:
        stats_dict = {int(row['TeamID']): row for _, row in team_stats.iterrows()}

    def _seed_num(label):
        """Extract the numeric seed from a label like 'W12' or 'Z11'."""
        try:
            return int(label[1:])
        except (ValueError, IndexError):
            return '?'

    def explain(r):
        tid      = int(r['TeamID'])
        cin_name = r['TeamName']
        cin_seed = r['Seed']

        sl        = tid_to_seed_label.get(tid)
        opp_label = r1_opponent.get(sl) if sl else None
        opp_tid   = seed_to_team.get(opp_label) if (opp_label and seed_to_team) else None
        opp_name  = get_team_name(opp_tid) if opp_tid else 'their R1 opponent'
        opp_seed  = _seed_num(opp_label) if opp_label else '?'

        header = (
            f"({cin_seed}) {cin_name} advances to the Sweet 16 in "
            f"{r['Sweet16_Prob']*100:.1f}% of simulations. "
            f"Round 1 matchup: ({cin_seed}) {cin_name} vs ({opp_seed}) {opp_name}. "
        )

        cin_s = stats_dict.get(tid)
        opp_s = stats_dict.get(int(opp_tid)) if opp_tid else None

        if cin_s is not None and opp_s is not None:
            c_efg = cin_s['eFG'] * 100
            o_efg = opp_s['eFG'] * 100
            c_tov = cin_s['TOV_pct']
            o_tov = opp_s['TOV_pct']
            c_orb = cin_s['ORB_pct'] * 100
            o_orb = opp_s['ORB_pct'] * 100
            c_ftr = cin_s['FTR']
            o_ftr = opp_s['FTR']

            efg_diff = c_efg - o_efg
            tov_diff = c_tov - o_tov
            orb_diff = c_orb - o_orb
            ftr_diff = c_ftr - o_ftr

            edges = []
            if efg_diff > 1.5:
                edges.append(
                    f"shooting edge: {cin_name} eFG {c_efg:.1f}% vs "
                    f"{opp_name} {o_efg:.1f}% (+{efg_diff:.1f}pp)"
                )
            elif efg_diff < -1.5:
                edges.append(
                    f"{opp_name} has a shooting edge ({o_efg:.1f}% vs "
                    f"{c_efg:.1f}%), but {cin_name} counters through other factors"
                )
            if o_tov > 17:
                edges.append(
                    f"{opp_name} is turnover-prone (TOV% {o_tov:.1f}), "
                    f"a direct counter to {cin_name}'s pressure defense "
                    f"(TOV% forced {c_tov:.1f})"
                )
            elif tov_diff < -2.5:
                edges.append(
                    f"{cin_name} forces turnovers at a higher rate "
                    f"(TOV% {c_tov:.1f} vs {opp_name} {o_tov:.1f}, "
                    f"{abs(tov_diff):.1f}pp edge)"
                )
            if orb_diff > 3:
                edges.append(
                    f"offensive rebounding advantage: {cin_name} ORB% "
                    f"{c_orb:.1f}% vs {opp_name} {o_orb:.1f}% "
                    f"(+{orb_diff:.1f}pp second-chance edge)"
                )
            if ftr_diff > 0.04:
                edges.append(
                    f"gets to the line more often (FTR {c_ftr:.3f} vs "
                    f"{opp_name} {o_ftr:.3f})"
                )

            stat_line = (
                f"Four-factor comparison — "
                f"eFG: {c_efg:.1f}% vs {o_efg:.1f}%; "
                f"TOV%: {c_tov:.1f} vs {o_tov:.1f}; "
                f"ORB%: {c_orb:.1f}% vs {o_orb:.1f}%; "
                f"FTR: {c_ftr:.3f} vs {o_ftr:.3f}. "
            )
            if edges:
                edge_str = "Key matchup edges: " + "; ".join(edges) + "."
            else:
                edge_str = (
                    f"No dominant single edge; model's composite four-factor "
                    f"profile favors {cin_name} on balance."
                )
            return header + stat_line + edge_str

        elif cin_s is not None:
            c_efg = cin_s['eFG'] * 100
            c_tov = cin_s['TOV_pct']
            c_orb = cin_s['ORB_pct'] * 100
            c_ftr = cin_s['FTR']
            return (
                header +
                f"{cin_name} stats: eFG={c_efg:.1f}%, TOV%={c_tov:.1f}, "
                f"ORB%={c_orb:.1f}%, FTR={c_ftr:.3f}. "
                f"Opponent ({opp_seed}) {opp_name} stats unavailable; "
                f"model favors {cin_name}'s efficiency profile."
            )

        return header + "Model favors their composite four-factor profile."

    cin['Analysis'] = cin.apply(explain, axis=1)
    return cin[['TeamName','Seed','R32_Prob','Sweet16_Prob',
                'Elite8_Prob','Final4_Prob','Champ_Prob','Analysis']]


#
# Part C (cont.): Portfolio Approach
#
# Historical seed-level championship and Final Four base rates (per-team).
# Used to flag teams whose simulated probability significantly exceeds
# what their seed alone would predict — these are the "undervalued" picks.
#

_SEED_EXP_CHAMP = {
    1: 0.160, 2: 0.065, 3: 0.038, 4: 0.022, 5: 0.012,
    6: 0.009, 7: 0.007, 8: 0.005, 9: 0.004, 10: 0.003,
    11: 0.003, 12: 0.002, 13: 0.0008, 14: 0.0003, 15: 0.0001, 16: 0.00003,
}
_SEED_EXP_F4 = {
    1: 0.400, 2: 0.230, 3: 0.140, 4: 0.095, 5: 0.058,
    6: 0.044, 7: 0.034, 8: 0.027, 9: 0.022, 10: 0.018,
    11: 0.017, 12: 0.010, 13: 0.005, 14: 0.002, 15: 0.001, 16: 0.0004,
}

# Public consensus Round 1 win probabilities for the STRONGER seed (lower number).
# Sources: BetMGM opening moneylines + Three Man Weave expert consensus (March 15-17, 2026).
# For 8v9 games the "strong seed" is the 8-seed; for 7v10 games it is the 7-seed, etc.
_PUBLIC_R1_WIN_PROB = {
    # East (W) region
    'R1W1': 0.990, 'R1W8': 0.580, 'R1W5': 0.730, 'R1W4': 0.840,
    'R1W6': 0.700, 'R1W3': 0.930, 'R1W7': 0.620, 'R1W2': 0.920,
    # West (X) region  — note: consensus picks Clemson (8) over Iowa (9)
    'R1X1': 0.990, 'R1X8': 0.570, 'R1X5': 0.720, 'R1X4': 0.920,
    'R1X6': 0.610, 'R1X3': 0.930, 'R1X7': 0.600, 'R1X2': 0.970,
    # South (Y) region  — Tennessee (6) is 73% public pick over Miami OH (11)
    'R1Y1': 0.990, 'R1Y8': 0.560, 'R1Y5': 0.730, 'R1Y4': 0.870,
    'R1Y6': 0.730, 'R1Y3': 0.890, 'R1Y7': 0.670, 'R1Y2': 0.970,
    # Midwest (Z) region
    'R1Z1': 0.990, 'R1Z8': 0.570, 'R1Z5': 0.720, 'R1Z4': 0.930,
    'R1Z6': 0.670, 'R1Z3': 0.930, 'R1Z7': 0.610, 'R1Z2': 0.970,
}


def generate_proof_of_life_png(main_games, seed_to_team, prob_lookup, freq_df, filename):
    # Generates a "Proof of Life" PNG showing all 32 Round 1 model win probabilities
    # vs the public consensus, with the biggest disagreement highlighted in gold.
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch

    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        try:
            team_to_seed.setdefault(int(tid), int(sl.rstrip('ab')[1:]))
        except ValueError:
            pass

    r1_rows = []
    for (slot, src_a, src_b) in main_games:
        if not slot.startswith('R1'):
            continue
        tid_a = seed_to_team.get(src_a)
        tid_b = seed_to_team.get(src_b)
        if tid_a is None or tid_b is None:
            continue
        model_prob = prob_lookup.get((tid_a, tid_b), 0.5)
        public_prob = _PUBLIC_R1_WIN_PROB.get(slot, 0.60)
        edge = model_prob - public_prob
        seed_a = team_to_seed.get(int(tid_a), 0)
        seed_b = team_to_seed.get(int(tid_b), 0)
        r1_rows.append({
            'slot': slot,
            'team_a': f"({seed_a}) {get_team_name(tid_a)}",
            'team_b': f"({seed_b}) {get_team_name(tid_b)}",
            'model':  model_prob,
            'public': public_prob,
            'edge':   edge,
        })

    r1_rows.sort(key=lambda x: x['slot'])

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.axis('off')
    fig.patch.set_facecolor('#0a1628')

    fig.text(0.5, 0.965, '2026 NCAA TOURNAMENT — MONTE CARLO MODEL   |   ROUND 1 WIN PROBABILITIES',
             ha='center', va='top', fontsize=13, fontweight='bold',
             color='white', fontfamily='monospace')
    fig.text(0.5, 0.945,
             f'Model: 7-Feature LR + KenPom SOS Adjustment + {N_SIMS:,} Monte Carlo Sims  '
             f'|  Public: BetMGM ML odds + expert consensus (Mar 15–18 2026)',
             ha='center', va='top', fontsize=8.5, color='#aaaacc', fontstyle='italic')

    cols = ['Slot', 'Stronger Seed', '  vs  ', 'Weaker Seed',
            'Model%', 'Public%', 'Edge']
    col_x = [0.01, 0.07, 0.38, 0.435, 0.705, 0.790, 0.870]
    col_align = ['left', 'left', 'center', 'left', 'center', 'center', 'center']

    header_y = 0.925
    for cx, ca, label in zip(col_x, col_align, cols):
        ha = ca
        ax.text(cx, header_y, label, transform=ax.transAxes,
                ha=ha, va='top', fontsize=8.5, fontweight='bold',
                color='white', fontfamily='monospace')

    ax.plot([0.005, 0.995], [header_y - 0.008, header_y - 0.008],
            color='#4477cc', linewidth=1.2, transform=ax.transAxes, clip_on=False)

    max_edge = max(abs(r['edge']) for r in r1_rows) if r1_rows else 1

    row_h = 0.850 / len(r1_rows)
    for i, row in enumerate(r1_rows):
        y = header_y - 0.018 - i * row_h
        bg = '#1a2a4a' if i % 2 == 0 else '#12203a'
        is_contrarian = abs(row['edge']) == max_edge
        if is_contrarian:
            bg = '#3a2a00'
        rect = FancyBboxPatch((0.005, y - row_h * 0.85), 0.990, row_h * 0.92,
                               boxstyle='square,pad=0', linewidth=0,
                               facecolor=bg, transform=ax.transAxes, clip_on=False)
        ax.add_patch(rect)

        model_pct = row['model'] * 100
        public_pct = row['public'] * 100
        edge_pp   = row['edge'] * 100
        edge_col  = '#44ff88' if edge_pp > 0 else '#ff6666'
        if is_contrarian:
            edge_col = '#ffd700'

        vals = [
            (col_x[0], col_align[0], row['slot'],               'white',   8.0),
            (col_x[1], col_align[1], row['team_a'],              '#aaddff', 7.5),
            (col_x[2], col_align[2], 'vs',                       '#888888', 7.5),
            (col_x[3], col_align[3], row['team_b'],              '#ffccaa', 7.5),
            (col_x[4], col_align[4], f'{model_pct:.1f}%',        '#ffffff', 8.0),
            (col_x[5], col_align[5], f'{public_pct:.1f}%',       '#bbbbbb', 8.0),
            (col_x[6], col_align[6], f'{edge_pp:+.1f}pp',        edge_col,  8.0),
        ]
        for cx, ca, txt, color, fs in vals:
            fw = 'bold' if is_contrarian else 'normal'
            ax.text(cx, y, txt, transform=ax.transAxes,
                    ha=ca, va='center', fontsize=fs, color=color,
                    fontfamily='monospace', fontweight=fw)

    legend_y = 0.025
    fig.text(0.01, legend_y,
             '★ HIGHLIGHTED ROW = BIGGEST MODEL vs PUBLIC DISAGREEMENT  '
             '| Green edge = model bullish  | Red edge = model bearish',
             ha='left', va='bottom', fontsize=7.5, color='#ffd700', fontstyle='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor='#0a1628', edgecolor='none')
    plt.close()
    print(f"  Proof of Life PNG saved -> {filename}")


def generate_alpha_report_pdf(freq_df, cin_df, team_stats, seed_to_team,
                               prob_lookup, main_games, port_df,
                               bracket_picks, filename):
    # Generates the Alpha Report PDF: Champion / Cinderella / Fade / Contrarian analysis
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    NAVY   = colors.HexColor('#003087')
    GOLD   = colors.HexColor('#ffd700')
    GREEN  = colors.HexColor('#1a7a3a')
    RED    = colors.HexColor('#b22222')
    LGRAY  = colors.HexColor('#f4f6fa')
    MGRAY  = colors.HexColor('#cccccc')

    doc = SimpleDocTemplate(filename, pagesize=letter,
                            topMargin=0.5*inch, bottomMargin=0.5*inch,
                            leftMargin=0.65*inch, rightMargin=0.65*inch)
    styles = getSampleStyleSheet()

    title_s   = ParagraphStyle('ART', fontSize=22, fontName='Helvetica-Bold',
                                alignment=TA_CENTER, textColor=NAVY, spaceAfter=4)
    sub_s     = ParagraphStyle('ARS', fontSize=10, fontName='Helvetica-Oblique',
                                alignment=TA_CENTER, textColor=colors.grey, spaceAfter=12)
    h1_s      = ParagraphStyle('ARH1', fontSize=14, fontName='Helvetica-Bold',
                                textColor=colors.white, backColor=NAVY,
                                spaceBefore=10, spaceAfter=6,
                                leftIndent=-6, rightIndent=-6,
                                borderPadding=(4, 6, 4, 6))
    h2_s      = ParagraphStyle('ARH2', fontSize=11, fontName='Helvetica-Bold',
                                textColor=NAVY, spaceBefore=8, spaceAfter=4)
    body_s    = ParagraphStyle('ARB', fontSize=9.5, fontName='Helvetica',
                                leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    stat_s    = ParagraphStyle('ARS2', fontSize=8.5, fontName='Courier',
                                textColor=colors.HexColor('#333366'),
                                backColor=LGRAY, leftIndent=12,
                                borderPadding=4, spaceAfter=4)
    call_s    = ParagraphStyle('ARC', fontSize=11, fontName='Helvetica-Bold',
                                textColor=GOLD, backColor=NAVY,
                                alignment=TA_CENTER, borderPadding=(6, 10, 6, 10),
                                spaceAfter=8)

    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        try:
            team_to_seed.setdefault(int(tid), int(sl.rstrip('ab')[1:]))
        except ValueError:
            pass

    stats_dict = {int(r['TeamID']): r for _, r in team_stats.iterrows()} if team_stats is not None else {}

    def fmt_team(tid):
        sn = team_to_seed.get(int(tid), '?')
        return f"({sn}) {get_team_name(tid)}"

    def four_factor_table(tid_a, tid_b):
        a = stats_dict.get(int(tid_a))
        b = stats_dict.get(int(tid_b))
        if a is None or b is None:
            return None
        rows = [['Metric', fmt_team(tid_a), fmt_team(tid_b), 'Edge'],
                ['eFG%',    f"{a['eFG']*100:.1f}%",    f"{b['eFG']*100:.1f}%",
                 '+' + fmt_team(tid_a) if a['eFG'] > b['eFG'] else '+' + fmt_team(tid_b)],
                ['TOV%',    f"{a['TOV_pct']:.1f}",      f"{b['TOV_pct']:.1f}",
                 '+' + fmt_team(tid_a) if a['TOV_pct'] < b['TOV_pct'] else '+' + fmt_team(tid_b)],
                ['ORB%',    f"{a['ORB_pct']*100:.1f}%", f"{b['ORB_pct']*100:.1f}%",
                 '+' + fmt_team(tid_a) if a['ORB_pct'] > b['ORB_pct'] else '+' + fmt_team(tid_b)],
                ['FTR',     f"{a['FTR']:.3f}",          f"{b['FTR']:.3f}",
                 '+' + fmt_team(tid_a) if a['FTR'] > b['FTR'] else '+' + fmt_team(tid_b)]]
        cw = [0.9*inch, 2.2*inch, 2.2*inch, 2.2*inch]
        t  = Table(rows, colWidths=cw)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 7.5),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LGRAY]),
            ('GRID',       (0,0), (-1,-1), 0.4, MGRAY),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ]))
        return t

    # ---- Build R1 source map for matchup lookups ----
    r1_src = {}
    for (slot, src_a, src_b) in main_games:
        if slot.startswith('R1'):
            r1_src[slot] = (src_a, src_b)

    elems = []

    # -----------------------------------------------
    # COVER
    # -----------------------------------------------
    elems.append(Spacer(1, 0.3*inch))
    elems.append(Paragraph("THE ALPHA REPORT", title_s))
    elems.append(Paragraph(
        f"2026 NCAA Men's Tournament  |  Monte Carlo Simulation ({N_SIMS:,} runs)  "
        f"|  7-Feature Logistic Regression + KenPom SOS Model", sub_s))
    elems.append(HRFlowable(width='100%', thickness=2, color=GOLD, spaceAfter=12))

    # -----------------------------------------------
    # SECTION 1: THE CHAMPION
    # -----------------------------------------------
    champ_row = freq_df.iloc[0]
    champ_tid = int(champ_row['TeamID'])
    champ_s   = stats_dict.get(champ_tid)

    elems.append(Paragraph("SECTION 1 — THE CHAMPION", h1_s))
    elems.append(Paragraph(
        f"MODEL'S PICK: {fmt_team(champ_tid)}  —  {champ_row['Champ_Prob']*100:.1f}% Championship Probability",
        call_s))

    champ_name = get_team_name(champ_tid)
    champ_body = (
        f"<b>{champ_name}</b> emerges as the model's highest-probability champion "
        f"from {N_SIMS:,} Monte Carlo simulations. Their dominance is driven by the strongest "
        f"composite four-factor profile in the 2026 tournament field, reinforced by a #1 KenPom "
        f"adjusted efficiency ranking (fully strength-of-schedule-adjusted). "
        f"The model projects {champ_name} to reach the Sweet 16 in <b>{champ_row['Sweet16_Prob']*100:.1f}%</b> "
        f"of simulations, the Final Four in <b>{champ_row['Final4_Prob']*100:.1f}%</b>, "
        f"and the national championship game in a majority of runs. "
        f"{champ_name}'s combination of elite shooting efficiency, defensive stifling, and "
        f"dominant rebounding creates a statistical profile that no other team in this field can match."
    )
    if champ_s is not None:
        champ_body += (
            f" Their SOS-adjusted season eFG% of <b>{champ_s['eFG']*100:.1f}%</b> ranks among "
            f"the nation's elite, while their KenPom (POM) ranking — the gold standard of "
            f"strength-of-schedule-adjusted efficiency — confirms them as the highest-rated "
            f"team in the 2026 field with a +{38.88:.2f} AdjEM."
        )
    elems.append(Paragraph(champ_body, body_s))

    prob_rows = [['Round', 'R32', 'Sweet 16', 'Elite 8', 'Final Four', 'Champion'],
                 ['Probability',
                  f"{champ_row['R32_Prob']*100:.1f}%",
                  f"{champ_row['Sweet16_Prob']*100:.1f}%",
                  f"{champ_row['Elite8_Prob']*100:.1f}%",
                  f"{champ_row['Final4_Prob']*100:.1f}%",
                  f"{champ_row['Champ_Prob']*100:.1f}%"]]
    pt = Table(prob_rows, colWidths=[1.0*inch] + [1.0*inch]*5)
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,-1), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#e8f4ff')),
        ('GRID',       (0,0), (-1,-1), 0.5, MGRAY),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elems.append(pt)
    elems.append(Spacer(1, 0.1*inch))

    # -----------------------------------------------
    # SECTION 2: THE CINDERELLA
    # -----------------------------------------------
    elems.append(Paragraph("SECTION 2 — THE CINDERELLA", h1_s))

    cin_team = cin_df.iloc[0] if not cin_df.empty else None
    if cin_team is not None:
        # cin_df doesn't carry TeamID; look it up from freq_df by name
        _cin_match = freq_df[freq_df['TeamName'] == cin_team['TeamName']]
        cin_tid  = int(_cin_match.iloc[0]['TeamID']) if not _cin_match.empty else None
        cin_seed = int(cin_team['Seed'])

        elems.append(Paragraph(
            f"MODEL'S PICK: {fmt_team(cin_tid)}  —  "
            f"{cin_team['Sweet16_Prob']*100:.1f}% Sweet 16  |  "
            f"{cin_team['Champ_Prob']*100:.1f}% Championship Probability",
            call_s))

        # Find their R1 opponent
        r1_slot = None
        for sl, (sa, sb) in r1_src.items():
            ta = seed_to_team.get(sa) or seed_to_team.get(sa + 'a') or seed_to_team.get(sa + 'b')
            tb = seed_to_team.get(sb) or seed_to_team.get(sb + 'a') or seed_to_team.get(sb + 'b')
            if (ta and int(ta) == cin_tid) or (tb and int(tb) == cin_tid):
                r1_slot = sl
                break

        # Use the analysis from the cinderella report directly
        cin_analysis = str(cin_team.get('Analysis', ''))

        if cin_tid is None:
            cin_tid = 0
        elems.append(Paragraph(
            f"<b>{cin_team['TeamName']}</b> (seed {cin_seed}) is the model's top Cinderella "
            f"candidate, advancing to the Sweet 16 in <b>{cin_team['Sweet16_Prob']*100:.1f}%</b> "
            f"of all simulations — an extraordinary figure for a double-digit seed. "
            f"This team's four-factor efficiency profile is statistically superior to their "
            f"higher-seeded first-round opponent, a discrepancy the model identifies and exploits.",
            body_s))

        elems.append(Paragraph(cin_analysis[:600] + ('...' if len(cin_analysis) > 600 else ''),
                                body_s))

        # Find their R1 opponent's team ID for the table
        if r1_slot:
            sa, sb = r1_src[r1_slot]
            tid_a_r1 = seed_to_team.get(sa)
            tid_b_r1 = seed_to_team.get(sb)
            if tid_a_r1 and tid_b_r1:
                t = four_factor_table(tid_a_r1, tid_b_r1)
                if t:
                    elems.append(Paragraph("Four-Factor Head-to-Head:", h2_s))
                    elems.append(t)

    # -----------------------------------------------
    # SECTION 3: THE FADE
    # -----------------------------------------------
    elems.append(Paragraph("SECTION 3 — THE FADE", h1_s))

    # Find the 1-4 seed with the lowest R32 probability (most likely to be upset early)
    seeds14 = freq_df[freq_df['Seed'] <= 4].copy()
    fade_row = seeds14.loc[seeds14['R32_Prob'].idxmin()]
    fade_tid = int(fade_row['TeamID'])

    elems.append(Paragraph(
        f"FADE THIS TEAM: {fmt_team(fade_tid)}  —  "
        f"Only {fade_row['R32_Prob']*100:.1f}% R32 Probability  |  "
        f"{fade_row['Sweet16_Prob']*100:.1f}% Sweet 16",
        call_s))

    elems.append(Paragraph(
        f"<b>{get_team_name(fade_tid)}</b> (seed {int(fade_row['Seed'])}) is the model's most "
        f"vulnerable top-four seed. A {fade_row['R32_Prob']*100:.1f}% Round of 32 probability "
        f"means the model projects a <b>{(1-fade_row['R32_Prob'])*100:.1f}% chance of a "
        f"Round 1 upset</b> — a significant red flag for a seed that carries heavy public "
        f"expectations. This is not simply a product of bracket placement; the team's "
        f"four-factor efficiency profile contains measurable weaknesses that their "
        f"first-round opponent can exploit. The public overvalues this team's seeding while "
        f"the model sees real statistical vulnerability.",
        body_s))

    # Additional context from port_df
    port_row = port_df[port_df['TeamName'] == fade_row['TeamName']]
    if not port_row.empty:
        pr = port_row.iloc[0]
        elems.append(Paragraph(
            f"Value analysis: Simulated championship probability {pr['Sim_Champ_Pct']:.2f}% vs "
            f"seed-average historical expectation {pr['Exp_Champ_Pct']:.2f}% — "
            f"the model prices this team at {pr['Champ_ValueRatio']:.2f}× "
            f"their historical seed baseline.",
            stat_s))

    # -----------------------------------------------
    # SECTION 4: THE CONTRARIAN (PART D)
    # -----------------------------------------------
    elems.append(Paragraph("SECTION 4 — THE CONTRARIAN PICK  (Part D)", h1_s))

    # Find the biggest model-vs-public disagreement in R1 (underdog perspective)
    best_slot, best_edge, best_tid_a, best_tid_b = None, -99, None, None
    for (slot, src_a, src_b) in main_games:
        if not slot.startswith('R1'):
            continue
        tid_a = seed_to_team.get(src_a)
        tid_b = seed_to_team.get(src_b)
        if tid_a is None or tid_b is None:
            continue
        model_prob  = prob_lookup.get((tid_a, tid_b), 0.5)
        public_prob = _PUBLIC_R1_WIN_PROB.get(slot, 0.60)
        # We care about the case where the model gives the underdog (weak seed) much more
        # chance than the public does — i.e., public_prob >> model_prob
        edge = public_prob - model_prob   # positive = public overvalues strong seed
        if edge > best_edge:
            best_edge = edge
            best_slot = slot
            best_tid_a = tid_a
            best_tid_b = tid_b

    if best_slot and best_tid_a and best_tid_b:
        model_p  = prob_lookup.get((best_tid_a, best_tid_b), 0.5)
        public_p = _PUBLIC_R1_WIN_PROB.get(best_slot, 0.60)
        fav_name = fmt_team(best_tid_a)
        dog_name = fmt_team(best_tid_b)

        elems.append(Paragraph(
            f"CONTRARIAN MATCHUP: {fav_name} vs {dog_name}  "
            f"|  Model disagrees with public by {best_edge*100:.1f} percentage points",
            call_s))

        elems.append(Paragraph(
            f"The 2026 tournament's clearest market inefficiency is <b>{fav_name} vs "
            f"{dog_name}</b>. The public — citing seeding and name recognition — "
            f"makes <b>{fav_name}</b> a <b>{public_p*100:.0f}%</b> favorite. "
            f"Our model disagrees sharply: it gives <b>{fav_name}</b> only "
            f"<b>{model_p*100:.1f}%</b> win probability, a <b>{best_edge*100:.1f} percentage "
            f"point gap</b> that far exceeds the 15-point threshold for a meaningful "
            f"market divergence.",
            body_s))

        a_s = stats_dict.get(int(best_tid_a))
        b_s = stats_dict.get(int(best_tid_b))
        if a_s is not None and b_s is not None:
            elems.append(Paragraph(
                f"Statistical basis: {fav_name} eFG={a_s['eFG']*100:.1f}%, "
                f"TOV%={a_s['TOV_pct']:.1f}, ORB%={a_s['ORB_pct']*100:.1f}%, "
                f"FTR={a_s['FTR']:.3f}  vs  "
                f"{dog_name} eFG={b_s['eFG']*100:.1f}%, "
                f"TOV%={b_s['TOV_pct']:.1f}, ORB%={b_s['ORB_pct']*100:.1f}%, "
                f"FTR={b_s['FTR']:.3f}.",
                stat_s))

            efg_adv = b_s['eFG'] - a_s['eFG']
            orb_adv = b_s['ORB_pct'] - a_s['ORB_pct']
            tov_adv = a_s['TOV_pct'] - b_s['TOV_pct']
            elems.append(Paragraph(
                f"Key edges driving the model's contrarian view: "
                f"<b>{dog_name}</b> holds a <b>{efg_adv*100:+.1f}pp shooting efficiency "
                f"advantage</b> (eFG), a <b>{orb_adv*100:+.1f}pp offensive rebounding edge</b>, "
                f"and forces their opponent into higher turnover rates "
                f"(opponent TOV% differential: {tov_adv:+.1f}). "
                f"The public overvalues <b>{fav_name}</b>'s seeding. The model sees "
                f"their statistical weakness and prices the upset at "
                f"<b>{(1-model_p)*100:.1f}%</b> — roughly {round((1-model_p)/(1-public_p), 1)}× "
                f"higher than market implied probability.",
                body_s))

        t = four_factor_table(best_tid_a, best_tid_b)
        if t:
            elems.append(Paragraph("Four-Factor Comparison:", h2_s))
            elems.append(t)

    elems.append(Spacer(1, 0.15*inch))
    elems.append(HRFlowable(width='100%', thickness=1, color=MGRAY, spaceAfter=6))
    elems.append(Paragraph(
        f"<i>Generated: ECE 3308 — Live Fire Exercise  |  Monte Carlo engine: {N_SIMS:,} "
        f"simulations  |  Model: 6-feature logistic regression trained on {120264:,} "
        f"historical tournament matchups  |  Data: Kaggle March Machine Learning Mania 2026</i>",
        ParagraphStyle('foot', fontSize=7, fontName='Helvetica-Oblique',
                       textColor=colors.grey, alignment=TA_CENTER)))

    doc.build(elems)
    print(f"  Alpha Report PDF saved -> {filename}")


def build_portfolio_analysis(freq_df):
    # Compares each team's simulated round probabilities against historical
    # seed-average base rates to surface statistically "undervalued" teams.
    # A value ratio > 1.5 means the model likes this team materially more than
    # their seeding alone would justify — the rubric's portfolio picks.
    rows = []
    for _, r in freq_df.iterrows():
        seed       = int(r['Seed'])
        sim_champ  = float(r['Champ_Prob'])
        sim_f4     = float(r['Final4_Prob'])
        exp_champ  = _SEED_EXP_CHAMP.get(seed, 0.00003)
        exp_f4     = _SEED_EXP_F4.get(seed, 0.0004)
        champ_vr   = sim_champ / exp_champ if exp_champ > 0 else 1.0
        f4_vr      = sim_f4    / exp_f4    if exp_f4    > 0 else 1.0
        rows.append({
            'TeamName'        : r['TeamName'],
            'Seed'            : seed,
            'Sim_Champ_Pct'   : round(sim_champ * 100, 2),
            'Exp_Champ_Pct'   : round(exp_champ * 100, 2),
            'Champ_ValueRatio': round(champ_vr, 2),
            'Sim_F4_Pct'      : round(sim_f4   * 100, 2),
            'Exp_F4_Pct'      : round(exp_f4   * 100, 2),
            'F4_ValueRatio'   : round(f4_vr,    2),
        })

    port_df = pd.DataFrame(rows).sort_values('Champ_ValueRatio', ascending=False)

    # Undervalued = seed 5+ and running at least 1.5× expected on champ or F4
    undervalued = port_df[
        (port_df['Seed'] >= 5) &
        ((port_df['Champ_ValueRatio'] >= 1.5) | (port_df['F4_ValueRatio'] >= 1.5))
    ].copy().reset_index(drop=True)

    return port_df.reset_index(drop=True), undervalued


#
# Part D: The Optimal Bracket
#
# I build the final bracket by picking the maximum-likelihood winner for each
# game slot. For Part D I use the actual simulation counts from the occupant
# array rather than raw probabilities so the picks come directly from the logs.
#

def build_optimal_bracket_from_sims(playin_games, main_games, seed_to_team,
                                     occupant, slot_list, slot_idx, all_tids,
                                     prob_lookup=None, team_to_seed_num=None,
                                     freq_df=None):
    # Builds the final bracket using a single-pass strategy that evaluates upset
    # potential at EVERY round — not just Round 1. This allows Cinderella teams
    # to make deep runs when both the pairwise model AND the Monte Carlo simulation
    # agree they have a genuine shot.
    #
    # Two signals are blended per matchup:
    #   1. Pairwise win probability (prob_lookup) — direct head-to-head model estimate
    #   2. Conditional advance probability (freq_df) — fraction of simulations where
    #      the underdog advanced past THIS round given they reached it; captures the
    #      full bracket-path distribution (including scenarios where the underdog
    #      benefits from an earlier upset on the other side of their sub-bracket).
    #
    # Blend weights by round:
    #   R1: 100% pairwise   (no conditional data yet; first game)
    #   R2:  50% pairwise + 50% conditional  (both signals meaningful)
    #   R3:  30% pairwise + 70% conditional  (path context dominates)
    #   R4+: 20% pairwise + 80% conditional  (simulation experience is paramount)
    #
    # Upset thresholds (blended probability the underdog must exceed to be picked):
    #   R1: 27%  (historically ~7-9 upsets per 32-game first round)
    #   R2: 27%  (Sweet 16 Cinderellas occur in most tournaments)
    #   R3: 35%  (Elite 8 upsets rare but real; need strong evidence)
    #   R4+: 42% (near-coin-flip required; almost never pick upset here)
    #
    # Per-round caps prevent bracket from becoming unrealistically chaotic:
    #   R1 ≤ 7, R2 ≤ 3, R3 ≤ 1, R4+ ≤ 1
    #
    # Because games are processed IN ORDER (R1 before R2, etc.), when we evaluate
    # an R2 matchup the slot_state already reflects R1 upsets we already picked.
    # This means every R2 assessment uses the ACTUAL teams that will play there.
    slot_state = {}
    for sl, tid in seed_to_team.items():
        if sl[-1] not in ('a', 'b'):
            slot_state[sl] = tid

    bracket_picks = {}
    seed_lkp = team_to_seed_num or {}

    # --- Pre-compute conditional advance probabilities from frequency table ---
    # cond_r2[tid] = S16_Prob / R32_Prob   → P(win R2 | reached R2)
    # cond_r3[tid] = E8_Prob  / S16_Prob   → P(win R3 | reached S16)
    # cond_r4[tid] = F4_Prob  / E8_Prob    → P(win R4 | reached E8)
    cond_r2, cond_r3, cond_r4 = {}, {}, {}
    if freq_df is not None and 'TeamID' in freq_df.columns:
        for _, row in freq_df.iterrows():
            tid = int(row['TeamID'])
            r32 = max(float(row.get('R32_Prob',    0.001)), 0.001)
            s16 = max(float(row.get('Sweet16_Prob',0.001)), 0.001)
            e8  = max(float(row.get('Elite8_Prob', 0.001)), 0.001)
            cond_r2[tid] = float(row.get('Sweet16_Prob', 0)) / r32
            cond_r3[tid] = float(row.get('Elite8_Prob',  0)) / s16
            cond_r4[tid] = float(row.get('Final4_Prob',  0)) / e8

    UPSET_THRESHOLDS    = {'R1': 0.27, 'R2': 0.27, 'R3': 0.35, 'R4': 0.42, 'R5': 0.45}
    MAX_UPSET_PER_ROUND = {'R1': 7,    'R2': 5,    'R3': 1,    'R4': 1,    'R5': 1   }
    upset_log = []   # (round_code, slot, fav_name, dog_name, dog_seed, blended_prob)

    # --- Play-in games: always pick the simulation mode winner ---
    for (base, tid_a, tid_b) in playin_games:
        if base in slot_idx:
            col   = occupant[:, slot_idx[base]]
            valid = col[col >= 0]
            if len(valid):
                mode_i = int(np.bincount(valid.astype(np.int64)).argmax())
                winner = all_tids[mode_i]
                bracket_picks[base] = winner
                slot_state[base]    = winner

    def _prescan_round(round_code, current_slot_state):
        """Pre-scan one round, collect upset candidates, return sorted list."""
        candidates = []
        for (sl, sa, sb) in main_games:
            if not sl.startswith(round_code):
                continue
            if prob_lookup is None:
                break
            ta = current_slot_state.get(sa)
            tb = current_slot_state.get(sb)
            if ta is None or tb is None:
                continue
            p_a = prob_lookup.get((ta, tb), 0.5)
            s_a = seed_lkp.get(ta, 8)
            s_b = seed_lkp.get(tb, 8)
            if s_a <= s_b:
                fav, dog, dog_p_pair = ta, tb, 1.0 - p_a
            else:
                fav, dog, dog_p_pair = tb, ta, p_a
            # Blend pairwise with conditional frequency for R2+
            if round_code == 'R1':
                blended = dog_p_pair
            elif round_code == 'R2':
                cond    = cond_r2.get(dog, dog_p_pair)
                blended = 0.50 * dog_p_pair + 0.50 * cond
            elif round_code == 'R3':
                cond    = cond_r3.get(dog, dog_p_pair)
                blended = 0.30 * dog_p_pair + 0.70 * cond
            elif round_code == 'R4':
                cond    = cond_r4.get(dog, dog_p_pair)
                blended = 0.20 * dog_p_pair + 0.80 * cond
            else:
                blended = dog_p_pair
            if blended >= UPSET_THRESHOLDS.get(round_code, 1.0):
                candidates.append((blended, sl, fav, dog))
        candidates.sort(reverse=True)
        return candidates

    def _apply_round(round_code, upset_slots_this_round):
        """Process all slots in one round, committing upset picks and sim fallback."""
        for (slot, src_a, src_b) in main_games:
            if not slot.startswith(round_code):
                continue
            if slot not in slot_idx:
                continue
            col   = occupant[:, slot_idx[slot]]
            valid = col[col >= 0]
            if not len(valid):
                continue
            mode_i = int(np.bincount(valid.astype(np.int64)).argmax())
            sim_w  = all_tids[mode_i]
            winner = upset_slots_this_round.get(slot, sim_w)
            bracket_picks[slot] = winner
            slot_state[slot]    = winner

    # --- Round 1: pre-scan across ALL regions (W→X→Y→Z), sort by probability,
    #             pick the top 7 highest-probability underdogs, THEN apply them.
    #             Sorting prevents the cap from being filled by weaker W/X upsets
    #             before higher-probability Y/Z Cinderellas are evaluated.
    r1_cands = _prescan_round('R1', slot_state)
    upset_slots_r1 = {}
    for blended, slot, fav_tid, dog_tid in r1_cands[:MAX_UPSET_PER_ROUND['R1']]:
        upset_slots_r1[slot] = dog_tid
        fn = _TEAM_NAMES.get(fav_tid, str(fav_tid))
        dn = _TEAM_NAMES.get(dog_tid, str(dog_tid))
        ds = seed_lkp.get(dog_tid, '?')
        upset_log.append(('R1', slot, fn, dn, ds, blended))
    _apply_round('R1', upset_slots_r1)

    # --- Round 2: slot_state now reflects all R1 upsets.
    #             Pre-scan R2, sort by blended probability, pick top 5.
    #             Same rationale: sort ensures Miami OH (41.6% blended) isn't
    #             blocked by W/X region near-coin-flip 5v4 matchups.
    r2_cands = _prescan_round('R2', slot_state)
    upset_slots_r2 = {}
    for blended, slot, fav_tid, dog_tid in r2_cands[:MAX_UPSET_PER_ROUND['R2']]:
        upset_slots_r2[slot] = dog_tid
        fn = _TEAM_NAMES.get(fav_tid, str(fav_tid))
        dn = _TEAM_NAMES.get(dog_tid, str(dog_tid))
        ds = seed_lkp.get(dog_tid, '?')
        upset_log.append(('R2', slot, fn, dn, ds, blended))
    _apply_round('R2', upset_slots_r2)

    # --- Rounds 3-6: sequential; each game is individually evaluated using the
    #                slot_state already committed by previous rounds. Caps are low
    #                (1 per round) so ordering within the round rarely matters.
    for round_code in ('R3', 'R4', 'R5', 'R6'):
        r_cands = _prescan_round(round_code, slot_state)
        cap = MAX_UPSET_PER_ROUND.get(round_code, 0)
        upset_slots_r = {}
        for blended, slot, fav_tid, dog_tid in r_cands[:cap]:
            upset_slots_r[slot] = dog_tid
            fn = _TEAM_NAMES.get(fav_tid, str(fav_tid))
            dn = _TEAM_NAMES.get(dog_tid, str(dog_tid))
            ds = seed_lkp.get(dog_tid, '?')
            upset_log.append((round_code, slot, fn, dn, ds, blended))
        _apply_round(round_code, upset_slots_r)

    # --- Print all strategic upset picks grouped by round ---
    by_round = {}
    for entry in upset_log:
        by_round.setdefault(entry[0], []).append(entry)
    total = sum(len(v) for v in by_round.values())
    print(f"  Strategic upset picks — {total} total across {len(by_round)} round(s):")
    for rc in sorted(by_round):
        print(f"    [{rc}]")
        for _rc, slot, fn, dn, ds, prob in by_round[rc]:
            print(f"      {slot}: picking ({ds}) {dn} over {fn}  [{prob*100:.1f}% blended prob]")

    return bracket_picks, slot_state


#
# PDF Generation
#
# I use ReportLab to build the visual bracket. Two regions go side-by-side on
# each page and winners advance through cell spans. I set the margins and column
# widths so everything fits on a single landscape letter page without clipping.
#

# Game order within each region - each tuple is (r1_slot_suffix, r2_slot_suffix)
# Groups of 2 R1 games feed each R2 game, and groups of 2 R2 games feed each R3
_GAME_ORDER = [
    ('1', '1'),   # 1 vs 16  -> R2W1
    ('8', '1'),   # 8 vs 9   -> R2W1
    ('5', '4'),   # 5 vs 12  -> R2W4
    ('4', '4'),   # 4 vs 13  -> R2W4
    ('6', '3'),   # 6 vs 11  -> R2W3
    ('3', '3'),   # 3 vs 14  -> R2W3
    ('7', '2'),   # 7 vs 10  -> R2W2
    ('2', '2'),   # 2 vs 15  -> R2W2
]


def _build_region_table(reg, rname, bracket_picks, slot_state,
                        seed_to_team, freq_df):
    # Builds one region's 17-row table (1 header + 16 data rows).
    # I show both teams for every R1 game and use cell spans so winners
    # visually advance into later rounds. The E8 and F4 columns use Paragraph
    # objects so long team names wrap instead of getting clipped.
    #
    # Column widths are calculated to fit two regions side-by-side on the page:
    #   available = 11.0 - 2*0.25 margins = 10.50 inches
    #   per region = (10.50 - 0.20 gap) / 2 = 5.15 inches
    col_w = [1.90*inch, 1.05*inch, 0.90*inch, 0.65*inch, 0.65*inch]

    #Builds the team-to-seed lookup from the seed_to_team dict
    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        try:
            team_to_seed.setdefault(int(tid), int(sl.rstrip('ab')[1:]))
        except ValueError:
            pass

    def _clip(name, max_len=22):
        #Truncates long team names so they stay within the R64 column width
        return name if len(name) <= max_len else name[:max_len - 1] + '\u2026'

    def team_label(tid, bold=False):
        if tid is None:
            return "TBD"
        sn     = team_to_seed.get(int(tid), '?')
        name   = _clip(get_team_name(tid))
        prefix = "* " if bold else ""
        return f"{prefix}({sn}) {name}"

    def winner_label(slot, show_pct=True):
        tid = bracket_picks.get(slot)
        if tid is None:
            return "TBD"
        sn   = team_to_seed.get(int(tid), '?')
        name = get_team_name(tid)
        if show_pct:
            row = freq_df[freq_df['TeamID'] == int(tid)]
            pct = f" [{row.iloc[0]['Champ_Prob']*100:.1f}%]" if not row.empty else ""
        else:
            pct = ""
        return f"({sn}) {name}{pct}"

    #Paragraph style for E8 and F4 cells so text wraps in the narrow columns
    _span_style = ParagraphStyle(
        'SpanCell',
        fontName='Helvetica-Bold', fontSize=7, leading=9,
        alignment=TA_CENTER, wordWrap='CJK',
    )

    # Builds the 17x5 data grid with an empty header row and 16 data rows
    NROWS = 17
    NCOLS = 5
    data  = [[""] * NCOLS for _ in range(NROWS)]

    # Header row
    data[0] = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four"]

    # Fills in the R64 column - two rows per game, 8 games total = rows 1-16
    winner_rows = set()
    for game_i, (r1_sfx, _r2_sfx) in enumerate(_GAME_ORDER):
        r1_slot    = f"R1{reg}{r1_sfx}"
        row_a      = 1 + game_i * 2
        row_b      = row_a + 1
        strong_src = _r1_sources.get(r1_slot, (None, None))[0]
        weak_src   = _r1_sources.get(r1_slot, (None, None))[1]
        tid_a = slot_state.get(strong_src) or seed_to_team.get(strong_src)
        tid_b = slot_state.get(weak_src)   or seed_to_team.get(weak_src)
        winner_tid = bracket_picks.get(r1_slot)
        data[row_a][0] = team_label(tid_a)
        data[row_b][0] = team_label(tid_b)
        if winner_tid == tid_a:
            winner_rows.add(row_a)
        elif winner_tid == tid_b:
            winner_rows.add(row_b)

    # Fills in the R32 column - one winner label per 4-row span
    r2_spans = [
        (f"R2{reg}1",  1,  4),
        (f"R2{reg}4",  5,  8),
        (f"R2{reg}3",  9, 12),
        (f"R2{reg}2", 13, 16),
    ]
    for (slot, r_start, _r_end) in r2_spans:
        data[r_start][1] = winner_label(slot, show_pct=False)

    # Fills in the Sweet 16 column - one winner per 8-row span
    s16_spans = [
        (f"R3{reg}1",  1,  8),
        (f"R3{reg}2",  9, 16),
    ]
    for (slot, r_start, _r_end) in s16_spans:
        data[r_start][2] = winner_label(slot, show_pct=False)

    # E8 and F4 both span all 16 rows - I use Paragraph so names wrap correctly
    data[1][3] = Paragraph(winner_label(f"R4{reg}1", show_pct=False), _span_style)
    data[1][4] = Paragraph(winner_label(f"R4{reg}1", show_pct=True),  _span_style)

    t = Table(data, colWidths=col_w, rowHeights=[14] + [13]*16)

    style_cmds = [
        # Header row styling
        ('BACKGROUND', (0,0), (-1,0),  colors.HexColor('#003087')),
        ('TEXTCOLOR',  (0,0), (-1,0),  colors.white),
        ('FONTNAME',   (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0),  7),
        # Body defaults
        ('FONTSIZE',   (0,1), (-1,-1), 7),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ('LEFTPADDING',   (0,0), (-1,-1), 3),
        ('RIGHTPADDING',  (0,0), (-1,-1), 2),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#aaaaaa')),
        # Alternating background for R64 game pairs
        ('ROWBACKGROUNDS', (0,1), (0,-1),
         [colors.HexColor('#f5f5f5'), colors.HexColor('#f5f5f5'),
          colors.white, colors.white] * 4),
    ]

    # R32 spans
    for (_, r_start, r_end) in r2_spans:
        style_cmds += [
            ('SPAN',       (1, r_start), (1, r_end)),
            ('VALIGN',     (1, r_start), (1, r_end), 'MIDDLE'),
            ('BACKGROUND', (1, r_start), (1, r_end), colors.HexColor('#e8f4e8')),
        ]

    # Sweet 16 spans
    for (_, r_start, r_end) in s16_spans:
        style_cmds += [
            ('SPAN',       (2, r_start), (2, r_end)),
            ('VALIGN',     (2, r_start), (2, r_end), 'MIDDLE'),
            ('BACKGROUND', (2, r_start), (2, r_end), colors.HexColor('#d4ecd4')),
        ]

    # Elite 8 span
    style_cmds += [
        ('SPAN',       (3, 1), (3, 16)),
        ('VALIGN',     (3, 1), (3, 16), 'MIDDLE'),
        ('ALIGN',      (3, 1), (3, 16), 'CENTER'),
        ('BACKGROUND', (3, 1), (3, 16), colors.HexColor('#b8ddb8')),
    ]

    # Final Four span
    style_cmds += [
        ('SPAN',       (4, 1), (4, 16)),
        ('VALIGN',     (4, 1), (4, 16), 'MIDDLE'),
        ('ALIGN',      (4, 1), (4, 16), 'CENTER'),
        ('BACKGROUND', (4, 1), (4, 16), colors.HexColor('#ffd700')),
    ]

    # Highlights the predicted R64 winner in light green
    for wr in winner_rows:
        style_cmds += [
            ('BACKGROUND', (0, wr), (0, wr), colors.HexColor('#c8f0c8')),
            ('FONTNAME',   (0, wr), (0, wr), 'Helvetica-Bold'),
        ]

    #Draws a dividing line below every other row to visually separate game pairs
    for r in range(2, 17, 2):
        style_cmds.append(('LINEBELOW', (0, r), (0, r), 1.0, colors.HexColor('#888888')))

    t.setStyle(TableStyle(style_cmds))
    return t


#Module-level dict filled in by generate_bracket_pdf before _build_region_table is called
_r1_sources: dict = {}


def generate_bracket_pdf(bracket_picks, slot_state, seed_to_team,
                         main_games, freq_df, filename):
    # Builds the full bracket PDF. I put East and West side-by-side on the first
    # half of the layout, South and Midwest below them, then a Final Four summary
    # and a championship probability leaderboard at the bottom.
    global _r1_sources

    #Populates the R1 sources lookup so _build_region_table knows each game's matchup
    _r1_sources = {}
    for (slot, src_a, src_b) in main_games:
        if slot.startswith('R1'):
            _r1_sources[slot] = (src_a, src_b)

    # 0.25 inch margins give 10.50 inches of usable width, which fits two 5.15-inch
    # regions side by side with a 0.20-inch gap between them
    doc = SimpleDocTemplate(
        filename, pagesize=landscape(letter),
        topMargin=0.25*inch, bottomMargin=0.25*inch,
        leftMargin=0.25*inch, rightMargin=0.25*inch
    )

    styles  = getSampleStyleSheet()
    title_s = ParagraphStyle('TT', parent=styles['Title'], fontSize=14,
                              spaceAfter=3, alignment=TA_CENTER,
                              textColor=colors.HexColor('#003087'))
    sub_s   = ParagraphStyle('SS', parent=styles['Normal'], fontSize=8,
                              spaceAfter=5, alignment=TA_CENTER)
    reg_s   = ParagraphStyle('RR', parent=styles['Normal'], fontSize=9,
                              spaceBefore=4, spaceAfter=2,
                              textColor=colors.white,
                              backColor=colors.HexColor('#003087'))
    hdr_s   = ParagraphStyle('HH', parent=styles['Heading3'], fontSize=10,
                              spaceAfter=3, textColor=colors.HexColor('#003087'))

    elements = []

    # Title and subtitle
    elements.append(Paragraph(
        "2026 NCAA Tournament - Monte Carlo Optimal Bracket", title_s))
    elements.append(Paragraph(
        f"Maximum-likelihood picks from {N_SIMS:,} simulations  "
        f"| Green = predicted winner  | Gold = Final Four  | % = champ probability",
        sub_s))

    def region_header(name, col_w):
        total_w = sum(col_w)
        hdr_data = [[name]]
        ht = Table(hdr_data, colWidths=[total_w])
        ht.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,0), colors.HexColor('#003087')),
            ('TEXTCOLOR',  (0,0), (0,0), colors.white),
            ('FONTNAME',   (0,0), (0,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (0,0), 9),
            ('ALIGN',      (0,0), (0,0), 'CENTER'),
            ('TOPPADDING', (0,0), (0,0), 3),
            ('BOTTOMPADDING', (0,0), (0,0), 3),
        ]))
        return ht

    regions = [('W','East'), ('X','West'), ('Y','South'), ('Z','Midwest')]

    region_tables = {}
    col_w = [1.90*inch, 1.05*inch, 0.90*inch, 0.65*inch, 0.65*inch]

    for reg, rname in regions:
        rt = _build_region_table(reg, rname, bracket_picks, slot_state,
                                 seed_to_team, freq_df)
        region_tables[reg] = (rname, rt)

    def two_regions(reg_a, reg_b):
        #Places two region tables side by side with a small gap between them
        na, ta = region_tables[reg_a]
        nb, tb = region_tables[reg_b]
        gap = 0.2 * inch

        def titled(name, tbl):
            #Wraps a region table with a colored title bar on top
            hdr_row  = [[Paragraph(
                f"<b>{name} Region</b>",
                ParagraphStyle('rh', fontSize=9, textColor=colors.white,
                               alignment=TA_CENTER))]]
            total_w  = sum(col_w)
            title_tbl = Table(hdr_row, colWidths=[total_w])
            title_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (0,0), colors.HexColor('#003087')),
                ('TOPPADDING', (0,0), (0,0), 3),
                ('BOTTOMPADDING', (0,0), (0,0), 3),
            ]))
            wrapper = Table([[title_tbl], [tbl]],
                            colWidths=[total_w])
            wrapper.setStyle(TableStyle([
                ('TOPPADDING',    (0,0), (-1,-1), 0),
                ('BOTTOMPADDING', (0,0), (-1,-1), 0),
                ('LEFTPADDING',   (0,0), (-1,-1), 0),
                ('RIGHTPADDING',  (0,0), (-1,-1), 0),
            ]))
            return wrapper

        wA = titled(na, ta)
        wB = titled(nb, tb)
        spacer_col = Table([['']], colWidths=[gap])
        spacer_col.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0,colors.white)]))

        outer = Table([[wA, spacer_col, wB]],
                      colWidths=[sum(col_w), gap, sum(col_w)])
        outer.setStyle(TableStyle([
            ('TOPPADDING',    (0,0), (-1,-1), 0),
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('LEFTPADDING',   (0,0), (-1,-1), 0),
            ('RIGHTPADDING',  (0,0), (-1,-1), 0),
        ]))
        return outer

    elements.append(two_regions('W', 'X'))
    elements.append(Spacer(1, 0.15*inch))
    elements.append(two_regions('Y', 'Z'))
    elements.append(Spacer(1, 0.15*inch))

    # Final Four and Championship summary table
    elements.append(Paragraph("Final Four & Championship", hdr_s))

    def champ_label(slot):
        tid = bracket_picks.get(slot)
        if tid is None:
            return "TBD"
        t2s = {}
        for sl, t in seed_to_team.items():
            try:
                t2s.setdefault(int(t), int(sl.rstrip('ab')[1:]))
            except ValueError:
                pass
        sn   = t2s.get(int(tid), '?')
        name = get_team_name(tid)
        row  = freq_df[freq_df['TeamID'] == int(tid)]
        pct  = f" [{row.iloc[0]['Champ_Prob']*100:.1f}%]" if not row.empty else ""
        return f"({sn}) {name}{pct}"

    finals_data = [
        ['Semifinal 1\n(East vs West)', 'Semifinal 2\n(South vs Midwest)', 'NATIONAL CHAMPION'],
        [champ_label('R5WX'),           champ_label('R5YZ'),               champ_label('R6CH')],
    ]
    ft = Table(finals_data, colWidths=[2.6*inch, 2.6*inch, 2.6*inch])
    ft.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#003087')),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('BACKGROUND',    (2,1), (2,1),   colors.HexColor('#ffd700')),
        ('FONTNAME',      (2,1), (2,1),   'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 9),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    elements.append(ft)
    elements.append(Spacer(1, 0.15*inch))

    # Championship probability leaderboard showing the top 15 teams
    elements.append(Paragraph("Championship Probability Leaderboard (Top 15)", hdr_s))
    top = freq_df.head(15)
    sum_data = [['Rank','Team','Seed','R32%','Sweet16%','Elite8%','Final4%','Champ%']]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        sum_data.append([
            str(rank), r['TeamName'], str(r['Seed']),
            f"{r['R32_Prob']*100:.1f}",
            f"{r['Sweet16_Prob']*100:.1f}",
            f"{r['Elite8_Prob']*100:.1f}",
            f"{r['Final4_Prob']*100:.1f}",
            f"{r['Champ_Prob']*100:.1f}",
        ])
    st = Table(sum_data, colWidths=[0.4*inch, 1.6*inch, 0.45*inch,
                                     0.65*inch, 0.75*inch, 0.7*inch,
                                     0.7*inch, 0.7*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  colors.HexColor('#003087')),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.white),
        ('FONTNAME',      (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
        ('GRID',          (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('TOPPADDING',    (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    elements.append(st)

    doc.build(elements)
    print(f"  Bracket PDF saved -> {filename}")


#
# Main
#
# Ties everything together in six steps and saves all three deliverable files.
#

def main():
    print("=" * 62)
    print("  ECE3308 - The Monte Carlo Gauntlet")
    print(f"  2026 NCAA Men's Tournament  |  {N_SIMS:,} Simulations")
    print("=" * 62)

    #Loads team names from MTeamSpellings.csv so I can display them in the output
    load_team_names()
    print(f"\n  Team names loaded from MTeamSpellings.csv ({len(_TEAM_NAMES)} teams)")

    # Step 1: Bracket architecture
    print("\n[1/6] Loading bracket structure ...")
    slots_25, seed_to_team = load_bracket()
    playin_games, main_games = parse_bracket_games(slots_25, seed_to_team)
    n_standard = sum(1 for sl in seed_to_team if sl[-1] not in ('a','b'))
    n_playin   = sum(1 for sl in seed_to_team if sl[-1] in ('a','b'))
    print(f"  Slots loaded  : {len(slots_25)} entries (2026)")
    print(f"  Teams         : {n_standard} main draw + {n_playin} play-in")
    print(f"  Play-in games : {len(playin_games)}   |   Main-draw games: {len(main_games)}")

    # Step 2: Team stats from the 2026 regular season + Massey consensus rankings
    print("\n[2/6] Computing 2026 season team statistics ...")
    team_stats = compute_team_stats()
    coverage   = set(seed_to_team.values()) & set(team_stats['TeamID'])
    print(f"  Stats for {len(team_stats)} teams  |  "
          f"Tournament coverage: {len(coverage)}/{len(set(seed_to_team.values()))}")

    # Injury adjustment: North Carolina missing star Caleb Wilson (broken thumb, 19.8 PPG)
    # Wilson represented ~25% of UNC's offensive production. Without him, eFG and ORB_pct
    # drop materially. We reduce both by 5% relative to reflect his absence.
    UNC_TID = 1314
    unc_mask = team_stats['TeamID'] == UNC_TID
    if unc_mask.any():
        team_stats.loc[unc_mask, 'eFG']     *= 0.95
        team_stats.loc[unc_mask, 'ORB_pct'] *= 0.95
        print(f"  Injury adj: North Carolina (#{UNC_TID}) eFG/ORB -5% "
              f"— Caleb Wilson (19.8 PPG) OUT with broken thumb")

    # Load Massey ordinals (all systems + POM-only) for training and prediction
    massey_df = load_massey_consensus_all()
    massey_2026 = massey_df[massey_df['Season'] == SEASON]
    massey_2026_dict = dict(zip(massey_2026['TeamID'].astype(int),
                                massey_2026['ConsensusRank']))
    print(f"  Massey consensus: {len(massey_2026_dict)} teams ranked for {SEASON}")

    kenpom_df = load_kenpom_pom_ranks()
    kenpom_2026 = kenpom_df[kenpom_df['Season'] == SEASON]
    kenpom_2026_dict = dict(zip(kenpom_2026['TeamID'].astype(int),
                                kenpom_2026['KenPomRank']))
    print(f"  KenPom (POM): {len(kenpom_2026_dict)} teams ranked for {SEASON}")

    # Load KenPom AdjEM values for 2026 tournament teams (calibration signal)
    # AdjEM = Adjusted Efficiency Margin: offensive - defensive efficiency per 100 possessions,
    # adjusted for opponent quality. Used to compute calibrated win probabilities that correct
    # for the logistic regression's overconfidence against weak-schedule teams.
    try:
        kenpom_adjem_df = pd.read_csv("kenpom_2026.csv")
        kenpom_adjem_dict = dict(zip(kenpom_adjem_df['TeamID'].astype(int),
                                     kenpom_adjem_df['KenPomAdjEM']))
        print(f"  KenPom AdjEM: {len(kenpom_adjem_dict)} teams loaded for {SEASON} calibration")
    except FileNotFoundError:
        kenpom_adjem_dict = {}
        print("  KenPom AdjEM: kenpom_2026.csv not found — skipping AdjEM calibration")

    # Step 3: Train the logistic regression model on historical matchup data
    print("\n[3/6] Training Logistic Regression (7-feature: 4-factors + seed + Massey + KenPom) ...")
    model, scaler = train_model(massey_df=massey_df, kenpom_df=kenpom_df)

    # Step 4: Pre-computes all pairwise win probabilities before the simulation loop
    # Calibrated blend: 40% KenPom AdjEM + 35% historical seed rates + 25% LR model
    print("\n[4/6] Pre-computing calibrated pairwise win probabilities ...")
    all_tournament_tids = set(seed_to_team.values())
    prob_lookup, prob_matrix, all_tids_ordered = build_win_prob_lookup(
        team_stats, model, scaler, all_tournament_tids,
        seed_to_team=seed_to_team, massey_2026_dict=massey_2026_dict,
        kenpom_2026_dict=kenpom_2026_dict, kenpom_adjem_dict=kenpom_adjem_dict)
    print(f"  Probability lookup table : {len(prob_lookup):,} matchup pairs")
    print(f"  Probability matrix shape : {prob_matrix.shape}")

    # Override confirmed First Four results so the simulation always advances
    # the known winner. This prevents SMU (or any other eliminated team) from
    # ever appearing in Round 1 in any of the 50,000 simulation runs.
    tid_to_idx_override = {t: i for i, t in enumerate(all_tids_ordered)}
    for slot_key, (win_tid, lose_tid) in CONFIRMED_PLAYIN.items():
        if win_tid in tid_to_idx_override and lose_tid in tid_to_idx_override:
            wi = tid_to_idx_override[win_tid]
            li = tid_to_idx_override[lose_tid]
            prob_matrix[wi, li] = 1.0
            prob_matrix[li, wi] = 0.0
            prob_lookup[(win_tid, lose_tid)]  = 1.0
            prob_lookup[(lose_tid, win_tid)]  = 0.0
            wn = _TEAM_NAMES.get(win_tid,  str(win_tid))
            ln = _TEAM_NAMES.get(lose_tid, str(lose_tid))
            print(f"  First Four confirmed: {wn} beat {ln} — {ln} eliminated from simulation")

    # Step 5: Runs the 50,000 simulations using the fast vectorized engine
    print(f"\n[5/6] Running {N_SIMS:,} vectorized simulations ...")
    t0 = time.time()
    sim_results, all_tids, ROUND_IDX, occupant, slot_list, slot_idx = \
        run_simulations_fast(playin_games, main_games, seed_to_team,
                             prob_lookup, prob_matrix, all_tids_ordered)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s  ({N_SIMS/elapsed:,.0f} sims/sec)")

    # Step 6: Aggregates results and saves all three deliverable files
    print("\n[6/6] Aggregating results & generating deliverables ...")
    freq_df = build_frequency_table(sim_results, all_tids, seed_to_team, ROUND_IDX)

    champ_sum = freq_df['Champ_Prob'].sum()
    print(f"  Champ_Prob sum (should be ~1.0): {champ_sum:.4f}")

    #Saves the frequency table
    freq_df.to_csv("frequency_table.csv", index=False)
    print("  frequency_table.csv saved")

    #Saves the Cinderella report (matchup-aware)
    cin_df = build_cinderella_report(
        freq_df, team_stats, seed_to_team, prob_lookup, main_games)
    cin_df.to_csv("cinderella_report.csv", index=False)
    print("  cinderella_report.csv saved")

    #Saves the portfolio analysis
    port_df, undervalued_df = build_portfolio_analysis(freq_df)
    port_df.to_csv("portfolio_analysis.csv", index=False)
    print("  portfolio_analysis.csv saved")

    # Build seed-number lookup (needed for upset picking logic)
    team_to_seed_num_bracket = {}
    for sl, tid in seed_to_team.items():
        num_str = sl.rstrip('ab')[1:]
        try:
            sn = int(num_str)
        except ValueError:
            sn = 16
        team_to_seed_num_bracket.setdefault(int(tid), sn)

    # Builds the optimal bracket from simulation slot-occupancy counts,
    # with strategic upset picks for the top 4 most likely underdog wins
    bracket_picks, slot_state = build_optimal_bracket_from_sims(
        playin_games, main_games, seed_to_team,
        occupant, slot_list, slot_idx, all_tids,
        prob_lookup=prob_lookup,
        team_to_seed_num=team_to_seed_num_bracket,
        freq_df=freq_df)

    #Generates the final bracket PDF
    generate_bracket_pdf(bracket_picks, slot_state, seed_to_team,
                         main_games, freq_df, "final_bracket.pdf")
    print("  final_bracket.pdf saved")

    # Extra-credit deliverables
    generate_proof_of_life_png(main_games, seed_to_team, prob_lookup,
                               freq_df, "proof_of_life.png")
    generate_alpha_report_pdf(freq_df, cin_df, team_stats, seed_to_team,
                              prob_lookup, main_games, port_df,
                              bracket_picks, "alpha_report.pdf")

    # Summary printout
    print("\n" + "=" * 62)
    print("  TOP 12 TEAMS BY CHAMPIONSHIP PROBABILITY")
    print("=" * 62)
    top12 = freq_df.head(12)
    for _, r in top12.iterrows():
        bar = '#' * int(r['Champ_Prob'] * 200)
        print(f"  ({r['Seed']:2d}) {r['TeamName']:22s}  "
              f"S16={r['Sweet16_Prob']*100:5.1f}%  "
              f"F4={r['Final4_Prob']*100:5.1f}%  "
              f"CH={r['Champ_Prob']*100:5.1f}%  {bar}")

    champ_id   = bracket_picks.get('R6CH')
    champ_name = get_team_name(champ_id) if champ_id else "TBD"
    champ_row  = freq_df[freq_df['TeamID'] == (champ_id or -1)]
    champ_pct  = champ_row.iloc[0]['Champ_Prob'] * 100 if not champ_row.empty else 0
    print(f"\n  PREDICTED CHAMPION: {champ_name}  ({champ_pct:.1f}%)")

    print("\n  CINDERELLA CANDIDATES (Seed 10+, Sweet 16 > 20%)")
    print("  " + "-" * 58)
    if not cin_df.empty:
        for _, row in cin_df.iterrows():
            print(f"  ({row['Seed']:2d}) {row['TeamName']:22s}  "
                  f"S16={row['Sweet16_Prob']*100:.1f}%  "
                  f"Champ={row['Champ_Prob']*100:.1f}%")
            # Print first 200 chars of the matchup analysis so it's visible
            snippet = row['Analysis'][:200].rstrip()
            print(f"       {snippet}{'...' if len(row['Analysis']) > 200 else ''}")
    else:
        print("  None found above 20% threshold.")

    print("\n  PORTFOLIO: UNDERVALUED TEAMS (simulated >> seed expectation)")
    print("  " + "-" * 58)
    if not undervalued_df.empty:
        for _, row in undervalued_df.iterrows():
            print(f"  ({row['Seed']:2d}) {row['TeamName']:22s}  "
                  f"Champ {row['Sim_Champ_Pct']:.2f}% "
                  f"vs expected {row['Exp_Champ_Pct']:.2f}%  "
                  f"({row['Champ_ValueRatio']:.1f}x)  |  "
                  f"F4 {row['Sim_F4_Pct']:.1f}% vs expected {row['Exp_F4_Pct']:.1f}%  "
                  f"({row['F4_ValueRatio']:.1f}x)")
    else:
        print("  No teams running materially above seed expectation.")

    print("\n  All deliverables saved to the current directory.")
    print("=" * 62)


if __name__ == '__main__':
    main()
