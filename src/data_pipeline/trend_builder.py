from .data_aggregator import DataAggregator
from .espn_client import ESPNClient
import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from pandas import DataFrame
from typing import Tuple

CONTEXT_WINDOW = 5
FEATURES = 15

class TrendBuilder:

    def __init__(self):
        DataAggregator.load_game_log()

    def get_player_trends(self, game_id: int, poi_id: str) -> Tensor:
        teammate_ids, opponent_ids = ESPNClient.get_player_ids(game_id, poi_id)
        poi_data = self.get_last_n_games(game_id, poi_id, CONTEXT_WINDOW)[0]

        teammate_data = torch.empty((len(teammate_ids), CONTEXT_WINDOW, FEATURES), 
                           dtype=torch.float32)
        for i, teammate_id in enumerate(teammate_ids):
            last_n = self.get_last_n_games(game_id=game_id, 
                                                     poi_id=teammate_id, 
                                                     n=CONTEXT_WINDOW)[0]
            print(last_n, last_n.shape[0] == 0)
            num_rows = last_n.shape[0]
            if num_rows < CONTEXT_WINDOW:
                last_n = F.pad(
                    last_n,
                    (0, 0, 0, CONTEXT_WINDOW - num_rows)
                )
                last_n[num_rows:,0] = 1
                print(last_n)
            teammate_data[i] = last_n
            
        opponent_data = torch.empty((len(opponent_ids), CONTEXT_WINDOW, FEATURES), 
                           dtype=torch.float32)
        for i, opponent_id in enumerate(opponent_ids):
            last_n = self.get_last_n_games(game_id=game_id, 
                                                     poi_id=opponent_id, 
                                                     n=CONTEXT_WINDOW)[0]
            print(last_n)
            num_rows = last_n.shape[0]
            if num_rows < CONTEXT_WINDOW:
                last_n = F.pad(
                    last_n,
                    (0, 0, 0, CONTEXT_WINDOW - num_rows)
                )
                last_n[num_rows:,0] = 1
            teammate_data[i] = last_n
        
        return poi_data, teammate_data, opponent_data

    
    def get_last_n_games(self, game_id: int, poi_id: int, n: int) -> Tuple[Tensor, DataFrame]:
        game_date = ESPNClient.get_game_date(game_id)
        last_n_before_game = (
            DataAggregator.game_log[(DataAggregator.game_log["GAME_DATE"] <= game_date) 
                                    & (DataAggregator.game_log["PLAYER_ID"] == poi_id)]
            .sort_values("GAME_DATE", ascending=False)
            .head(n)
        )
        data = tensor(last_n_before_game.iloc[:, 3:].values, dtype=torch.float32)
        metadata = last_n_before_game.iloc[:, :3]
        return data, metadata