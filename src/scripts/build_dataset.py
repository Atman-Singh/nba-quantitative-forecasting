import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.raw_data_aggregator import RawDataAggregator
from data_pipeline.dataset_builder import DatasetBuilder

def main():
    trend_builder = DatasetBuilder()
    trend_builder.build_dataset()
    # print(DataAggregator.game_log)
    # trend_builder.build_player_trends(game_id=401873343, poi_id=4222252)[1]
    

if __name__ == "__main__":
    main()