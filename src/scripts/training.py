import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.espn_client import ESPNClient
from data_pipeline.data_aggregator import DataAggregator

def main():
    poi = '4701230'
    matchups = ESPNClient.get_last_n_matchups(1, 2, 5)
    # DataAggregator.build_player_game_logs()
    gl = DataAggregator.load_game_log()
    print(gl)


if __name__ == "__main__":
    main()