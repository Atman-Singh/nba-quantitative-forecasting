import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.dataset_builder import DatasetBuilder
from utils.timing import timed

RUNS = 100

def main():
    db = DatasetBuilder()
    times = np.array([])
    for _ in range(RUNS):
        times = np.append(times, 
                  timed(db.build_player_trends)(game_id=401873343, 
                                                poi_id=4222252)[1])
    print(f"build_player_trends took ~{np.median(times)}s")

if __name__ == "__main__":
    main()