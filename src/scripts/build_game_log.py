import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.raw_data_aggregator import RawDataAggregator
from data_pipeline.dataset_builder import DatasetBuilder

def main():
    # RawDataAggregator.build_game_log(years=5, reload=True, save_step=100)
    RawDataAggregator.load_game_log()
    print(RawDataAggregator.load_player_ids())

if __name__ == "__main__":
    main()