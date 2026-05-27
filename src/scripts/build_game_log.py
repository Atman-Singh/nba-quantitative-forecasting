import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.raw_data_aggregator import RawDataAggregator
from data_pipeline.dataset_builder import DatasetBuilder

def main():
    RawDataAggregator.build_player_game_logs(years=5, save_step=20)

if __name__ == "__main__":
    main()