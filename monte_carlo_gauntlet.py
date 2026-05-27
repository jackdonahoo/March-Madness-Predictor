"""
ECE3308 - The Monte Carlo Gauntlet
2025 NCAA Tournament Monte Carlo Simulation (10,000 runs)

Architecture:
  Part A: Bracket Architecture  — builds slot structure from MNCAATourneySlots.csv
  Part B: Simulation Engine     — vectorized 10,000-run simulation using NumPy
  Part C: Aggregation & Analysis— per-team round probabilities, Cinderella report
  Part D: Optimal Bracket       — maximum-likelihood bracket construction

Deliverables:
  frequency_table.csv    (TeamName, Seed, R32_Prob…Champ_Prob)
  cinderella_report.csv  (seed 10+ teams with >20% Sweet 16)
  final_bracket.pdf      (visual 63-game bracket)
"""

import sys
import io
# Force UTF-8 output on Windows to handle Unicode characters in print()
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

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SEASON      = 2025
N_SIMS      = 10_000
RANDOM_SEED = 42
DATA_DIR    = "."
P1_DIR      = "../Project_1"

FEATURE_COLS = ['DIFF_EFG', 'Diff_TOV', 'DIFF_ORB', 'DIFF_FTR']
TARGET_COL   = 'Win_A'

# Round label produced when a team WINS that round's game
ROUND_WIN_LABEL = {
    'R1': 'R32',
    'R2': 'Sweet16',
    'R3': 'Elite8',
    'R4': 'Final4',
    'R5': 'Final_SF',
    'R6': 'Champion',
}

# Best-effort TeamID → team name  (from standard Kaggle NCAAM IDs)
TEAM_NAMES = {
    1103: "Akron",             1104: "Auburn",          1106: "Army",
    1110: "AR-Little Rock",    1112: "Alabama",          1116: "Arkansas",
    1120: "Arizona",           1124: "Arizona State",    1136: "Bellarmine",
    1140: "BYU",               1155: "Clemson",          1161: "Coastal Carolina",
    1163: "Colorado",          1166: "UConn",            1179: "Drake",
    1181: "Duke",              1188: "E Tennessee St",   1196: "Florida",
    1208: "Georgia",           1211: "Grand Canyon",     1213: "Georgia Southern",
    1219: "High Point",        1222: "Houston",          1228: "Illinois",
    1235: "Iowa State",        1242: "Kansas",           1246: "Kentucky",
    1251: "Liberty",           1252: "Lipscomb",         1257: "Loyola Chicago",
    1266: "Marquette",         1268: "Maryland",         1270: "McNeese",
    1272: "Memphis",           1276: "Michigan",         1277: "Michigan State",
    1279: "Minnesota",         1280: "Mississippi",      1281: "Missouri",
    1285: "Montana",           1291: "Nebraska",         1303: "North Iowa",
    1307: "Oakland",           1313: "Oregon",           1314: "Oregon State",
    1328: "Penn State",        1332: "Purdue",           1345: "Saint Mary's",
    1352: "San Diego State",   1361: "South Carolina",   1384: "St. Peter's",
    1385: "Syracuse",          1388: "Ohio State",       1397: "Tennessee",
    1400: "Texas",             1401: "Texas A&M",        1403: "Texas Tech",
    1407: "Toledo",            1417: "Utah State",       1423: "Vanderbilt",
    1429: "Vermont",           1433: "Villanova",        1435: "Virginia",
    1437: "VCU",               1458: "Wisconsin",        1459: "Wofford",
    1462: "Wright State",      1463: "Xavier",           1471: "Yale",
}


def get_team_name(tid):
    return TEAM_NAMES.get(int(tid), f"Team_{tid}")


# ─────────────────────────────────────────────────────────────────────────────
# PART A  —  BRACKET ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

def load_bracket():
    """
    Returns:
      slots_25  : DataFrame — 2025 rows from MNCAATourneySlots.csv
      seed_to_team : dict   — seed_label → TeamID  (e.g. 'W01' → 1181)
    """
    slots_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySlots.csv")
    seeds_df = pd.read_csv(f"{DATA_DIR}/MNCAATourneySeeds.csv")
    slots_25 = slots_df[slots_df['Season'] == SEASON].copy().reset_index(drop=True)
    seeds_25 = seeds_df[seeds_df['Season'] == SEASON].copy()
    seed_to_team = dict(zip(seeds_25['Seed'], seeds_25['TeamID'].astype(int)))
    return slots_25, seed_to_team


def parse_bracket_games(slots_25, seed_to_team):
    """
    Convert the slots DataFrame into an ordered list of game tuples:
       (slot_name, source_a, source_b)
    where source is either a seed label ('W01') or a prior slot name ('R1W1').

    Also identifies play-in games separately.
    Skips play-in definition rows (slot names like 'W16', 'X11' that are
    not real-round games but point to sub-seeds).
    """
    playin_games = []    # (slot_base, tid_a, tid_b) — e.g. ('W16', 1110, 1291)
    main_games   = []    # (slot_name, src_a, src_b)

    # Identify play-in seed labels
    playin_bases = set()
    for seed_label in seed_to_team:
        if seed_label[-1] in ('a', 'b'):
            playin_bases.add(seed_label[:-1])   # e.g. 'W16', 'X11'

    # Build play-in game pairs
    for base in playin_bases:
        label_a = base + 'a'
        label_b = base + 'b'
        tid_a = seed_to_team.get(label_a)
        tid_b = seed_to_team.get(label_b)
        if tid_a and tid_b:
            playin_games.append((base, tid_a, tid_b))

    # Parse main rounds from slots file
    for _, row in slots_25.iterrows():
        slot = row['Slot']
        if not slot.startswith('R'):
            continue            # skip play-in definition rows
        main_games.append((slot, row['StrongSeed'], row['WeakSeed']))

    return playin_games, main_games


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  —  2025 season stats
# ─────────────────────────────────────────────────────────────────────────────

def compute_team_stats():
    """
    Compute per-team 2025 season average four-factor stats from
    MRegularSeasonDetailedResults.csv using the exact same formulas
    as ECE3308_Project1.ipynb.

    Returns DataFrame: TeamID | eFG | TOV_pct | ORB_pct | FTR
    """
    rs = pd.read_csv(f"{P1_DIR}/MRegularSeasonDetailedResults.csv")
    rs = rs[rs['Season'] == SEASON].copy()

    # Winners perspective
    w = rs[['WTeamID','WFGM','WFGA','WFGM3','WFTA','WOR','LOR','WTO']].copy()
    w.columns = ['TeamID','FGM','FGA','FGM3','FTA','OR','oOR','TO']

    # Losers perspective
    l = rs[['LTeamID','LFGM','LFGA','LFGM3','LFTA','LOR','WOR','LTO']].copy()
    l.columns = ['TeamID','FGM','FGA','FGM3','FTA','OR','oOR','TO']

    all_rows = pd.concat([w, l], ignore_index=True)

    # Four factors (per-game)
    all_rows['eFG']     = (all_rows['FGM'] + 0.5 * all_rows['FGM3']) / all_rows['FGA']
    poss                = all_rows['FGA'] - all_rows['OR'] + all_rows['TO'] + 0.475 * all_rows['FTA']
    all_rows['TOV_pct'] = (all_rows['TO'] / poss.replace(0, np.nan)) * 100
    total_reb           = all_rows['OR'] + all_rows['oOR']
    all_rows['ORB_pct'] = all_rows['OR'] / total_reb.replace(0, np.nan)
    all_rows['FTR']     = all_rows['FTA'] / all_rows['FGA'].replace(0, np.nan)

    stats = (all_rows.groupby('TeamID')[['eFG','TOV_pct','ORB_pct','FTR']]
                     .mean()
                     .reset_index())
    return stats


def build_win_prob_lookup(team_stats, model, scaler, all_team_ids):
    """
    Pre-compute pairwise win probabilities for all possible matchups
    among the 68 tournament teams. Returns a dict (tid_a, tid_b) → prob_a_wins.
    
    This vectorized pre-computation eliminates repeated model calls
    inside the simulation hot loop.
    """
    ids = list(all_team_ids)
    n   = len(ids)
    idx_map = {tid: i for i, tid in enumerate(ids)}

    # Build feature matrix for all ordered pairs
    stats_dict = {int(row['TeamID']): row for _, row in team_stats.iterrows()}

    pairs_X = []
    pairs_key = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tid_a = ids[i]
            tid_b = ids[j]
            a = stats_dict.get(tid_a)
            b = stats_dict.get(tid_b)
            if a is None or b is None:
                pairs_X.append([0.0, 0.0, 0.0, 0.0])
            else:
                pairs_X.append([
                    a['eFG']     - b['eFG'],
                    a['TOV_pct'] - b['TOV_pct'],
                    a['ORB_pct'] - b['ORB_pct'],
                    a['FTR']     - b['FTR'],
                ])
            pairs_key.append((tid_a, tid_b))

    X = np.array(pairs_X)
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]

    lookup = {}
    for (tid_a, tid_b), p in zip(pairs_key, probs):
        lookup[(tid_a, tid_b)] = float(p)

    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """Train LogisticRegression on all historical tournament matchup data."""
    mat = pd.read_csv(f"{DATA_DIR}/Master_Analytical_Table.csv")
    mat = mat.dropna(subset=FEATURE_COLS + [TARGET_COL])

    X  = mat[FEATURE_COLS].values
    y  = mat[TARGET_COL].values

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_sc, y)

    acc = (model.predict(X_sc) == y).mean()
    print(f"  Model trained on {len(mat):,} matchups  |  train accuracy: {acc:.4f}")
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PART B  —  VECTORIZED SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_simulations(playin_games, main_games, seed_to_team, prob_lookup):
    """
    Run N_SIMS tournament simulations fully vectorized with NumPy.

    Strategy:
      - Represent each simulation's state as a dict: slot_name → team_idx
        (using integer indices into a teams array for speed).
      - For each game in round order, generate N_SIMS random numbers at once
        and compare to the precomputed probability to determine winners.
      - Track the furthest round each team reached.

    Returns:
      sim_results : np.ndarray, shape (N_SIMS, n_teams)
                    Each cell is an integer round index (0=PlayIn_Loss, 1=R64..6=Champion)
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # Team index mapping
    all_tids = sorted(set(seed_to_team.values()))
    tid_to_idx = {t: i for i, t in enumerate(all_tids)}
    n_teams = len(all_tids)

    # Round codes  (index)
    ROUND_IDX = {
        'PlayIn_Loss': 0,
        'R64'        : 1,
        'R32'        : 2,
        'Sweet16'    : 3,
        'Elite8'     : 4,
        'Final4'     : 5,
        'Final_SF'   : 6,
        'Champion'   : 7,
    }
    ROUND_WIN_IDX = {
        'R1': ROUND_IDX['R32'],
        'R2': ROUND_IDX['Sweet16'],
        'R3': ROUND_IDX['Elite8'],
        'R4': ROUND_IDX['Final4'],
        'R5': ROUND_IDX['Final_SF'],
        'R6': ROUND_IDX['Champion'],
    }

    # sim_results[sim, team_idx] = best round index reached
    sim_results = np.ones((N_SIMS, n_teams), dtype=np.int8)   # 1 = R64

    # slot_occupant[sim, slot_id] = team_idx in that slot for that simulation
    # We build a flat slot list first
    all_slots = {}   # slot_name → slot_id (integer)

    # Register all seed labels as slots
    for sl in seed_to_team:
        if sl not in all_slots:
            all_slots[sl] = len(all_slots)
    # Register all game result slots
    for (slot, src_a, src_b) in main_games:
        if slot not in all_slots:
            all_slots[slot] = len(all_slots)

    n_slots = len(all_slots)
    slot_occupant = np.full((N_SIMS, n_slots), -1, dtype=np.int32)

    # Initialize seed slots
    for sl, tid in seed_to_team.items():
        if sl[-1] not in ('a', 'b'):
            sid = all_slots[sl]
            slot_occupant[:, sid] = tid_to_idx[tid]

    # ── Play-in games (vectorized) ──────────────────────────────────────────
    for (base, tid_a, tid_b) in playin_games:
        if base not in all_slots:
            all_slots[base] = len(all_slots)
            # Resize slot_occupant if needed (rarely triggered)
            new_col = np.full((N_SIMS, 1), -1, dtype=np.int32)
            slot_occupant = np.hstack([slot_occupant, new_col])

        base_sid  = all_slots[base]
        idx_a = tid_to_idx[tid_a]
        idx_b = tid_to_idx[tid_b]
        prob  = prob_lookup.get((tid_a, tid_b), 0.5)

        rand  = rng.random(N_SIMS)
        a_wins = rand < prob

        # Winners enter the main draw (slot = base)
        slot_occupant[:, base_sid] = np.where(a_wins, idx_a, idx_b)

        # Losers are eliminated (PlayIn_Loss = 0)
        sim_results[a_wins,  idx_b] = 0
        sim_results[~a_wins, idx_a] = 0

    # ── Main rounds (vectorized) ────────────────────────────────────────────
    for (slot, src_a, src_b) in main_games:
        slot_sid = all_slots[slot]
        prefix   = slot[:2]
        round_won_idx = ROUND_WIN_IDX.get(prefix, 1)

        # Resolve source teams for all simulations
        if src_a in all_slots:
            teams_a = slot_occupant[:, all_slots[src_a]]
        else:
            # src_a is a seed label that might not have its own slot entry
            tid_a_fixed = seed_to_team.get(src_a)
            teams_a = np.full(N_SIMS, tid_to_idx.get(tid_a_fixed, -1), dtype=np.int32)

        if src_b in all_slots:
            teams_b = slot_occupant[:, all_slots[src_b]]
        else:
            tid_b_fixed = seed_to_team.get(src_b)
            teams_b = np.full(N_SIMS, tid_to_idx.get(tid_b_fixed, -1), dtype=np.int32)

        # For each simulation, look up win probability
        # Vectorized: build array of probs for each simulation
        # Most games have the same two teams (early rounds), so we can batch

        # Get unique (team_a, team_b) pairs and compute probs in batch
        pairs = np.stack([teams_a, teams_b], axis=1)
        unique_pairs = np.unique(pairs, axis=0)

        prob_cache = {}
        for pa, pb in unique_pairs:
            if pa == -1 or pb == -1:
                prob_cache[(pa, pb)] = 0.5
                continue
            tid_a2 = all_tids[pa]
            tid_b2 = all_tids[pb]
            prob_cache[(pa, pb)] = prob_lookup.get((tid_a2, tid_b2), 0.5)

        # Map each simulation's pair to its probability
        probs_vec = np.array([prob_cache[(a, b)] for a, b in
                               zip(teams_a.tolist(), teams_b.tolist())])

        rand = rng.random(N_SIMS)
        a_wins = (rand < probs_vec) & (teams_a != -1) & (teams_b != -1)

        winners = np.where(a_wins, teams_a, teams_b)
        losers  = np.where(a_wins, teams_b, teams_a)

        # Update deepest round for winners
        for sim_i in range(N_SIMS):
            w = winners[sim_i]
            if w >= 0:
                if sim_results[sim_i, w] < round_won_idx:
                    sim_results[sim_i, w] = round_won_idx

        # Place winners into this slot for use by next round
        slot_occupant[:, slot_sid] = winners

    return sim_results, all_tids, ROUND_IDX


def run_simulations_fast(playin_games, main_games, seed_to_team, prob_lookup):
    """
    Faster version using pure numpy with pre-allocated arrays.
    Avoids per-sim Python loops by processing all N_SIMS in parallel.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    all_tids  = sorted(set(seed_to_team.values()))
    tid_to_idx = {t: i for i, t in enumerate(all_tids)}
    n_teams   = len(all_tids)

    ROUND_WIN_IDX = {
        'R1': 2, 'R2': 3, 'R3': 4,
        'R4': 5, 'R5': 6, 'R6': 7,
    }
    # 0=PlayIn_Loss, 1=R64(entered), 2=R32, 3=S16, 4=E8, 5=F4, 6=SF, 7=Champ

    sim_results = np.ones((N_SIMS, n_teams), dtype=np.int8)

    # Build slot arrays: all_slot_names ordered list, occupant matrix
    slot_list = []
    slot_idx  = {}

    def get_slot(name):
        if name not in slot_idx:
            slot_idx[name] = len(slot_list)
            slot_list.append(name)
        return slot_idx[name]

    # Pre-register all slots to know final size
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

    n_slots = len(slot_list)
    occupant = np.full((N_SIMS, n_slots), -1, dtype=np.int16)

    # Initialise standard seed slots
    for sl, tid in seed_to_team.items():
        if sl[-1] not in ('a', 'b'):
            occupant[:, get_slot(sl)] = tid_to_idx[tid]

    # ── Play-in ─────────────────────────────────────────────────────────────
    for (base, tid_a, tid_b) in playin_games:
        sid   = get_slot(base)
        ia, ib = tid_to_idx[tid_a], tid_to_idx[tid_b]
        prob  = prob_lookup.get((tid_a, tid_b), 0.5)
        r     = rng.random(N_SIMS)
        a_wins = r < prob
        occupant[:, sid] = np.where(a_wins, ia, ib)
        sim_results[a_wins,  ib] = 0
        sim_results[~a_wins, ia] = 0

    # ── Main rounds ──────────────────────────────────────────────────────────
    for (slot, src_a, src_b) in main_games:
        out_sid     = get_slot(slot)
        round_won_v = ROUND_WIN_IDX.get(slot[:2], 1)

        # Resolve teams for each sim
        if src_a in slot_idx:
            ta_vec = occupant[:, slot_idx[src_a]].copy()
        else:
            fixed = tid_to_idx.get(seed_to_team.get(src_a, None), -1)
            ta_vec = np.full(N_SIMS, fixed, dtype=np.int16)

        if src_b in slot_idx:
            tb_vec = occupant[:, slot_idx[src_b]].copy()
        else:
            fixed = tid_to_idx.get(seed_to_team.get(src_b, None), -1)
            tb_vec = np.full(N_SIMS, fixed, dtype=np.int16)

        # Build probability vector (one prob per sim)
        # Unique pairs approach to minimise cache lookups
        stacked = np.stack([ta_vec, tb_vec], axis=1)
        unique_pairs, inv = np.unique(stacked, axis=0, return_inverse=True)
        pair_probs = np.empty(len(unique_pairs))
        for k, (pa, pb) in enumerate(unique_pairs):
            if pa < 0 or pb < 0:
                pair_probs[k] = 0.5
            else:
                pair_probs[k] = prob_lookup.get(
                    (all_tids[pa], all_tids[pb]), 0.5)
        prob_vec = pair_probs[inv]

        # Simulate all N_SIMS games at once
        r      = rng.random(N_SIMS)
        a_wins = (r < prob_vec) & (ta_vec >= 0) & (tb_vec >= 0)

        winners = np.where(a_wins, ta_vec, tb_vec).astype(np.int16)
        losers  = np.where(a_wins, tb_vec, ta_vec).astype(np.int16)

        # Update sim_results for winners (vectorized)
        # We need: sim_results[sim, winner] = max(current, round_won_v)
        # Use np.maximum.at  (scatter-max is not native in numpy,
        #   but we can do it safely with a loop only over unique winner values)
        valid = winners >= 0
        sim_results[np.where(valid)[0], winners[valid]] = np.maximum(
            sim_results[np.where(valid)[0], winners[valid]],
            round_won_v
        )

        occupant[:, out_sid] = winners

    ROUND_IDX = {
        'PlayIn_Loss': 0, 'R64': 1, 'R32': 2, 'Sweet16': 3,
        'Elite8': 4, 'Final4': 5, 'Final_SF': 6, 'Champion': 7,
    }
    return sim_results, all_tids, ROUND_IDX


# ─────────────────────────────────────────────────────────────────────────────
# PART C  —  AGGREGATION & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def build_frequency_table(sim_results, all_tids, seed_to_team, ROUND_IDX):
    """
    Build frequency table with round probabilities for every team.
    Returns DataFrame sorted by Champ_Prob descending.
    """
    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        num_str = sl.rstrip('ab')[1:]
        try:
            sn = int(num_str)
        except ValueError:
            sn = 99
        team_to_seed.setdefault(int(tid), sn)

    # Threshold indices
    R32_min  = ROUND_IDX['R32']
    S16_min  = ROUND_IDX['Sweet16']
    E8_min   = ROUND_IDX['Elite8']
    F4_min   = ROUND_IDX['Final4']
    CH_exact = ROUND_IDX['Champion']

    rows = []
    for i, tid in enumerate(all_tids):
        col = sim_results[:, i]
        # A team "reached" a round if its best result is >= that round's index
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
                             prob_lookup=None):
    """
    Find seed >= 10 teams with Sweet16_Prob > 20% (Cinderella candidates).
    Falls back to top-5 seed 10+ teams if threshold not met.
    Generates feature-based analysis explaining WHY the model likes them.
    """
    cin = freq_df[(freq_df['Seed'] >= 10) & (freq_df['Sweet16_Prob'] > 0.20)].copy()
    if cin.empty:
        cin = freq_df[freq_df['Seed'] >= 10].nlargest(5, 'Sweet16_Prob').copy()

    def explain(r):
        base_msg = (
            f"Seed {r['Seed']} team '{r['TeamName']}' advances to the Sweet 16 "
            f"in {r['Sweet16_Prob']*100:.1f}% of sims ({N_SIMS:,} total). "
            f"R32 probability: {r['R32_Prob']*100:.1f}%. "
        )
        # Add feature context if stats available
        if team_stats is not None:
            tid = r['TeamID']
            ts = team_stats[team_stats['TeamID'] == tid]
            if not ts.empty:
                t = ts.iloc[0]
                efg_pct  = t['eFG'] * 100
                tov_pct  = t['TOV_pct']
                orb_pct  = t['ORB_pct'] * 100
                ftr_val  = t['FTR']
                feat_msg = (
                    f"Key stats: eFG={efg_pct:.1f}%, TOV%={tov_pct:.1f}, "
                    f"ORB%={orb_pct:.1f}%, FTR={ftr_val:.3f}. "
                )
                context = ""
                if efg_pct > 53:
                    context += "Elite shooting efficiency. "
                if tov_pct < 14:
                    context += "Very low turnover rate (ball security advantage). "
                if orb_pct > 32:
                    context += "Strong offensive rebounding creates second chances. "
                if ftr_val > 0.35:
                    context += "High FTR means they draw fouls and get to the line. "
                return base_msg + feat_msg + context
        return base_msg + "Model favors their offensive efficiency profile vs. their early-round opponent."

    cin['Analysis'] = cin.apply(explain, axis=1)
    return cin[['TeamName','Seed','R32_Prob','Sweet16_Prob',
                'Elite8_Prob','Final4_Prob','Champ_Prob','Analysis']]


# ─────────────────────────────────────────────────────────────────────────────
# PART D  —  OPTIMAL BRACKET  (maximum-likelihood)
# ─────────────────────────────────────────────────────────────────────────────

def build_optimal_bracket(playin_games, main_games, seed_to_team, prob_lookup):
    """
    Walk through the bracket slot by slot.
    For each game, pick the team with win probability ≥ 0.50.
    Returns dict: slot_name → winner TeamID
    """
    all_tids   = sorted(set(seed_to_team.values()))
    tid_to_idx = {t: i for i, t in enumerate(all_tids)}

    # Working state: slot_name → TeamID
    slot_state = {}
    for sl, tid in seed_to_team.items():
        if sl[-1] not in ('a', 'b'):
            slot_state[sl] = tid

    # Resolve play-in
    for (base, tid_a, tid_b) in playin_games:
        prob = prob_lookup.get((tid_a, tid_b), 0.5)
        slot_state[base] = tid_a if prob >= 0.5 else tid_b

    picks = {}
    for (slot, src_a, src_b) in main_games:
        ta = slot_state.get(src_a) or seed_to_team.get(src_a)
        tb = slot_state.get(src_b) or seed_to_team.get(src_b)
        if ta is None or tb is None:
            continue
        prob = prob_lookup.get((ta, tb), 0.5)
        winner = ta if prob >= 0.5 else tb
        picks[slot]       = winner
        slot_state[slot]  = winner

    return picks


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_bracket_pdf(bracket_picks, seed_to_team, freq_df, filename):
    doc = SimpleDocTemplate(
        filename, pagesize=landscape(letter),
        topMargin=0.3*inch, bottomMargin=0.3*inch,
        leftMargin=0.35*inch, rightMargin=0.35*inch
    )

    styles   = getSampleStyleSheet()
    title_st = ParagraphStyle('T', parent=styles['Title'],
                               fontSize=15, spaceAfter=4,
                               alignment=TA_CENTER, textColor=colors.darkblue)
    sub_st   = ParagraphStyle('S', parent=styles['Normal'],
                               fontSize=9, spaceAfter=6,
                               alignment=TA_CENTER)
    hdr_st   = ParagraphStyle('H', parent=styles['Heading3'],
                               fontSize=10, spaceAfter=3,
                               textColor=colors.darkblue)

    team_to_seed = {}
    for sl, tid in seed_to_team.items():
        try:
            sn = int(sl.rstrip('ab')[1:])
        except ValueError:
            sn = 99
        team_to_seed.setdefault(int(tid), sn)

    def label(slot, show_pct=True):
        tid = bracket_picks.get(slot)
        if tid is None:
            return "TBD"
        sn   = team_to_seed.get(int(tid), '?')
        name = get_team_name(tid)
        if show_pct:
            row  = freq_df[freq_df['TeamID'] == int(tid)]
            pct  = f"  [{row.iloc[0]['Champ_Prob']*100:.1f}%]" if not row.empty else ""
        else:
            pct = ""
        return f"({sn}) {name}{pct}"

    elements = []
    elements.append(Paragraph("2025 NCAA Tournament — Monte Carlo Optimal Bracket", title_st))
    elements.append(Paragraph(
        f"Maximum-Likelihood picks based on {N_SIMS:,} Logistic Regression simulations  |  ECE3308",
        sub_st))
    elements.append(Spacer(1, 0.1*inch))

    # ── Regional brackets ────────────────────────────────────────────────────
    regions = [
        ('W', 'East'),  ('X', 'West'),
        ('Y', 'South'), ('Z', 'Midwest'),
    ]

    for reg, rname in regions:
        elements.append(Paragraph(f"{rname} Region", hdr_st))
        hdr = ['R1 (Round of 64)', 'R2 (Sweet 16)', 'R3 (Elite 8)',
               'R4 (Final Four)', 'Notes']
        data = [hdr]

        r1_slots = [f"R1{reg}{i}" for i in range(1, 9)]
        r2_slots = [f"R2{reg}{i}" for i in range(1, 5)]
        r3_slots = [f"R3{reg}{i}" for i in range(1, 3)]
        r4_slot  = f"R4{reg}1"

        for i, r1s in enumerate(r1_slots):
            r2s = r2_slots[i // 2]
            r3s = r3_slots[i // 4]
            row = [
                label(r1s, show_pct=False),
                label(r2s, show_pct=False) if i % 2 == 0 else '',
                label(r3s, show_pct=False) if i % 4 == 0 else '',
                label(r4_slot) if i == 0 else '',
                '',
            ]
            data.append(row)

        col_w = [1.9*inch, 1.9*inch, 1.9*inch, 1.9*inch, 1.4*inch]
        t = Table(data, colWidths=col_w)
        t.setStyle(TableStyle([
            ('BACKGROUND',     (0,0), (-1,0), colors.HexColor('#003087')),
            ('TEXTCOLOR',      (0,0), (-1,0), colors.white),
            ('FONTNAME',       (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',       (0,0), (-1,-1), 8),
            ('ALIGN',          (0,0), (-1,-1), 'LEFT'),
            ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
            ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
            ('TOPPADDING',     (0,0), (-1,-1), 3),
            ('BOTTOMPADDING',  (0,0), (-1,-1), 3),
            ('LEFTPADDING',    (0,0), (-1,-1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 0.12*inch))

    # ── Final Four & Championship ─────────────────────────────────────────
    elements.append(Paragraph("Final Four & Championship", hdr_st))

    finals_data = [
        ['Semifinal 1 (WX)',   'Semifinal 2 (YZ)',   'CHAMPION'],
        [label('R5WX'), label('R5YZ'), label('R6CH')],
    ]
    ft = Table(finals_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
    ft.setStyle(TableStyle([
        ('BACKGROUND',  (0,0), (-1,0), colors.HexColor('#003087')),
        ('TEXTCOLOR',   (0,0), (-1,0), colors.white),
        ('FONTNAME',    (0,0), (-1,0), 'Helvetica-Bold'),
        ('BACKGROUND',  (2,1), (2,1), colors.HexColor('#ffd700')),
        ('FONTNAME',    (2,1), (2,1), 'Helvetica-Bold'),
        ('FONTSIZE',    (0,0), (-1,-1), 9),
        ('ALIGN',       (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
        ('GRID',        (0,0), (-1,-1), 0.5, colors.grey),
        ('TOPPADDING',  (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    elements.append(ft)
    elements.append(Spacer(1, 0.15*inch))

    # ── Championship probability leaderboard ─────────────────────────────
    elements.append(Paragraph("Championship Probability Leaderboard (Top 15)", hdr_st))
    top = freq_df.head(15)
    sum_hdr = ['Rank','Team','Seed','R32%','Sweet16%','Elite8%','Final4%','Champ%']
    sum_data = [sum_hdr]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        sum_data.append([
            str(rank), r['TeamName'], str(r['Seed']),
            f"{r['R32_Prob']*100:.1f}",
            f"{r['Sweet16_Prob']*100:.1f}",
            f"{r['Elite8_Prob']*100:.1f}",
            f"{r['Final4_Prob']*100:.1f}",
            f"{r['Champ_Prob']*100:.1f}",
        ])
    st = Table(sum_data, colWidths=[0.4*inch, 1.55*inch, 0.4*inch,
                                     0.6*inch, 0.7*inch, 0.65*inch,
                                     0.65*inch, 0.65*inch])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor('#003087')),
        ('TEXTCOLOR',  (0,0),(-1,0), colors.white),
        ('FONTNAME',   (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0),(-1,-1), 8),
        ('ALIGN',      (0,0),(-1,-1), 'CENTER'),
        ('ROWBACKGROUNDS', (0,1),(-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
        ('GRID',       (0,0),(-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0,0),(-1,-1), 2),
        ('BOTTOMPADDING', (0,0),(-1,-1), 2),
    ]))
    elements.append(st)

    doc.build(elements)
    print(f"  Bracket PDF saved → {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  ECE3308 – The Monte Carlo Gauntlet")
    print(f"  2025 NCAA Men's Tournament  |  {N_SIMS:,} Simulations")
    print("=" * 62)

    # ── 1. Bracket architecture ─────────────────────────────────────────────
    print("\n[1/6] Loading bracket structure …")
    slots_25, seed_to_team = load_bracket()
    playin_games, main_games = parse_bracket_games(slots_25, seed_to_team)
    n_standard = sum(1 for sl in seed_to_team if sl[-1] not in ('a','b'))
    n_playin   = sum(1 for sl in seed_to_team if sl[-1] in ('a','b'))
    print(f"  Slots loaded  : {len(slots_25)} entries (2025)")
    print(f"  Teams         : {n_standard} main draw + {n_playin} play-in")
    print(f"  Play-in games : {len(playin_games)}   |   Main-draw games: {len(main_games)}")

    # ── 2. Team stats ────────────────────────────────────────────────────────
    print("\n[2/6] Computing 2025 season team statistics …")
    team_stats = compute_team_stats()
    coverage   = set(seed_to_team.values()) & set(team_stats['TeamID'])
    print(f"  Stats for {len(team_stats)} teams  |  "
          f"Tournament coverage: {len(coverage)}/{len(set(seed_to_team.values()))}")

    # ── 3. Train model ───────────────────────────────────────────────────────
    print("\n[3/6] Training Logistic Regression …")
    model, scaler = train_model()

    # ── 4. Pre-compute win probabilities ────────────────────────────────────
    print("\n[4/6] Pre-computing pairwise win probabilities …")
    all_tournament_tids = set(seed_to_team.values())
    prob_lookup = build_win_prob_lookup(team_stats, model, scaler,
                                        all_tournament_tids)
    print(f"  Probability lookup table: {len(prob_lookup):,} matchup pairs")

    # ── 5. Run 10,000 simulations ────────────────────────────────────────────
    print(f"\n[5/6] Running {N_SIMS:,} vectorized simulations …")
    t0 = time.time()
    sim_results, all_tids, ROUND_IDX = run_simulations_fast(
        playin_games, main_games, seed_to_team, prob_lookup)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.2f}s  ({N_SIMS/elapsed:,.0f} sims/sec)")

    # ── 6. Aggregate & generate deliverables ─────────────────────────────────
    print("\n[6/6] Aggregating results & generating deliverables …")
    freq_df = build_frequency_table(sim_results, all_tids, seed_to_team, ROUND_IDX)

    champ_sum = freq_df['Champ_Prob'].sum()
    print(f"  Champ_Prob sum (should ~1.0): {champ_sum:.4f}")

    # Frequency table
    freq_df.to_csv("frequency_table.csv", index=False)
    print("  ✔ frequency_table.csv")

    # Cinderella report
    cin_df = build_cinderella_report(freq_df, team_stats, seed_to_team, prob_lookup)
    cin_df.to_csv("cinderella_report.csv", index=False)
    print("  ✔ cinderella_report.csv")

    # Optimal bracket
    bracket_picks = build_optimal_bracket(playin_games, main_games,
                                           seed_to_team, prob_lookup)

    # PDF
    generate_bracket_pdf(bracket_picks, seed_to_team, freq_df, "final_bracket.pdf")
    print("  ✔ final_bracket.pdf")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  TOP 12 TEAMS BY CHAMPIONSHIP PROBABILITY")
    print("=" * 62)
    top12 = freq_df.head(12)
    for _, r in top12.iterrows():
        bar = '█' * int(r['Champ_Prob'] * 200)
        print(f"  ({r['Seed']:2d}) {r['TeamName']:22s}  "
              f"S16={r['Sweet16_Prob']*100:5.1f}%  "
              f"F4={r['Final4_Prob']*100:5.1f}%  "
              f"CH={r['Champ_Prob']*100:5.1f}%  {bar}")

    champ_id   = bracket_picks.get('R6CH')
    champ_name = get_team_name(champ_id) if champ_id else "TBD"
    champ_row  = freq_df[freq_df['TeamID'] == (champ_id or -1)]
    champ_pct  = champ_row.iloc[0]['Champ_Prob'] * 100 if not champ_row.empty else 0
    print(f"\n   PREDICTED CHAMPION: {champ_name}  ({champ_pct:.1f}%)")

    print("\n  CINDERELLA CANDIDATES (Seed ≥ 10, Sweet16 > 20%)")
    print("  " + "─" * 58)
    if not cin_df.empty:
        for _, row in cin_df.iterrows():
            print(f"  ({row['Seed']:2d}) {row['TeamName']:22s}  "
                  f"S16={row['Sweet16_Prob']*100:.1f}%  "
                  f"Champ={row['Champ_Prob']*100:.1f}%")
    else:
        print("  None found above 20% threshold.")

    print("\n  All deliverables saved to Project_2/")
    print("=" * 62)


if __name__ == '__main__':
    main()
