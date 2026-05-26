import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.data_aggregator import DataAggregator
from data_pipeline.trend_builder import TrendBuilder

def main():
    # poi = '4701230'
    # matchups = ESPNClient.get_last_n_matchups(1, 2, 5)
    # DataAggregator.build_player_game_logs()

    trend_builder = TrendBuilder()
    # print(DataAggregator.game_log)
    print(trend_builder.build_player_trends(game_id=401873343, poi_id=4222252)[1])
    

if __name__ == "__main__":
    main()