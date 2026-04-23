import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.espn_client import ESPNClient

def main():
    espn_client = ESPNClient()
    print(espn_client.get_last_n_games(1, 7))
    print(espn_client.get_last_n_matchups(1, 2, 5))

if __name__ == "__main__":
    main()