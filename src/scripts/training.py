import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.espn_client import ESPNClient
from data.data_pipeline import DataPipeline

def main():
    poi = '4701230'
    matchups = ESPNClient.get_last_n_matchups(1, 2, 5)
    print(DataPipeline.build_player_game_logs())

if __name__ == "__main__":
    main()