"""直接运行 generate_initial_routes.py。这里只设置计算规模，不需要命令行模式。"""
import os
from pathlib import Path

from settings import SEARCH_RUNS, SEARCH_THREADS, SEARCH_SEED
ROOT = Path(__file__).resolve().parents[2]
TARGET_DATA_FILE = ROOT / 'src/data/states.npy'
BEAM_RUNS = SEARCH_RUNS
BEAM_WIDTHS = (96, 192, 192, 288)
MAX_REVOLUTIONS_BY_RUN = (0, 0, 0, 1)
MIN_TARGET_GAP_S = 30.0  # 数值保护，不是竞赛规则；二阶段允许进一步缩短至10秒
SEARCH_COST_CEILING_KM_S = 24.0
REPRESENTATIVES_PER_BRANCH = 3
LOCAL_REFINEMENTS_PER_BRANCH = 1
GOLDEN_ITERATIONS = 5
TIME_DISTANCE_S = 600.0
VELOCITY_DISTANCE_KM_S = 0.06
EXPLORATION_FRACTION = 0.30
LOOKAHEAD_POOL_MULTIPLIER = 2
LOOKAHEAD_WEIGHT = 0.60
TIME_PENALTIES = (0.4, 0.8, 1.2, 1.6)
WORKERS = SEARCH_THREADS
RANDOM_SEED = SEARCH_SEED

# Plane-window variant: one uniformly sampled ORIENTED plane per beam run.
# No plane scoring, screening, old solutions or best-plane preferences.
PLANE_TOLERANCE_DEG = 6.0
WINDOW_EDGE_TOLERANCE_S = 0.05
# A wide/near-coplanar window is subdivided; it is never discarded.
WINDOW_CELL_MAX_H = 6.0
MAX_TRANSFER_H = 72.0

# Spend more effort on the first visit, within this run's sampled plane.
FIRST_MAX_ARRIVAL_H = 72.0
FIRST_PHASE_SAMPLES = 64
FIRST_TIME_SAMPLES_PER_CELL = 3
FIRST_REFINE_PER_TARGET = 4
FIRST_GOLDEN_ITERATIONS = 8
FIRST_COORDINATE_PASSES = 1
INITIAL_ALTITUDE_MIN_KM = 590.1
INITIAL_ALTITUDE_MAX_KM = 609.9

FIRST_ALTITUDE_SAMPLES = 2

FIRST_KEEP_PER_CELL_BRANCH = 4

FIRST_CELL_MAX_H = 6.0

EXPANSION_TIME_POINTS = 1

EXPANSION_MINIMA_PER_CELL = 1

STATE_BUCKET_CAP = 8

SEARCH_PROFILE = 'local_plane_fast'

LOOKAHEAD_CELL_MAX_H = 6.0

# Includes initialization/JIT. Completed route shards are committed per run.
SERVER_BUDGET_SECONDS = 0.0  # Local search stops after SEARCH_RUNS complete planes.
SERVER_MIN_RUN_RESERVE_SECONDS = 90.0
SERVER_RUN_RESERVE_FACTOR = 3.0

# Uniform coverage of the full oriented-normal sphere.
PLANE_SECTOR_INDEX = 1
PLANE_SECTOR_COUNT = 1
PLANE_NORMAL_Z_MIN = -1.0
PLANE_NORMAL_Z_MAX = 1.0
# Distinct streams even when the caller explicitly supplies the same base seed.
RANDOM_SEED += 1000003 * (PLANE_SECTOR_INDEX - 1)
