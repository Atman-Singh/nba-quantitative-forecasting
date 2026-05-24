from data_aggregator import DataAggregator
from espn_client import ESPNClient

CONTEXT_WINDOW = 5

class TrendBuilder:
    @staticmethod
    def get_player_trends(game_id: int, poi_id: str):
        teammate_ids, opponent_ids = ESPNClient.get_player_ids(game_id, poi_id)
        poi_last_n = ESPNClient.get_last_n_games(poi_id, CONTEXT_WINDOW)