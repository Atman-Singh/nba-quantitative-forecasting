from .espn_client import ESPNClient
import numpy as np
from torch import tensor

CONTEXT_WINDOW = 5

class DataPipeline():
    
    @staticmethod
    def build_player_game_logs():
        date = ESPNClient.get_current_date()
        for _ in range(50):
            date = ESPNClient.decrement_date(date)
            games = ESPNClient.get_scoreboard(date)['events']
            for game in games:
                game_id = game['id']
                box_scores = ESPNClient.get_box_scores(game_id)
                for team in box_scores:
                    for player in team['statistics'][0]['athletes']:
                        box_score = ESPNClient.format_box_score(player['stats'])
                


        
    @staticmethod
    def get_player_trends(game_id: int, poi_id: str):
        teammate_ids, opponent_ids = ESPNClient.get_player_ids(game_id, poi_id)
        poi_last_n = ESPNClient.get_last_n_games(poi_id, CONTEXT_WINDOW)
        