import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.dataset_builder import DatasetBuilder
from utils.timing import timed

RUNS = 1

def main():
    db = DatasetBuilder()
    times = np.array([])
    for _ in range(RUNS):
        times = np.append(times, 
                  timed(db.build_dataset)(years=3)[1])
    print(f"build_player_trends took ~{np.median(times)}s")
    print(f'trend cache accessed {db.trend_cache_accesses / db.trend_total * 100}% of game log queries')

if __name__ == "__main__":
    main()