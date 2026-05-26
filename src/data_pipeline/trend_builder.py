from .data_aggregator import DataAggregator
from .espn_client import ESPNClient
import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from pandas import DataFrame
from typing import Tuple

CONTEXT_WINDOW = 5
FEATURES = 17

class TrendBuilder:

    def __init__(self):
        DataAggregator.load_game_log()

    def build_player_trends(self, game_id: int, poi_id: int) -> Tensor:
        game_date = ESPNClient.get_game_date(game_id)
        poi_data = self.get_last_n_games(game_date, poi_id, CONTEXT_WINDOW)[0]

        player_ids = ESPNClient.get_player_ids(game_id, poi_id)
        num_players = max(len(player_ids[0]), len(player_ids[0]))
        data = torch.empty((2, num_players, CONTEXT_WINDOW, FEATURES), 
                            dtype=torch.float32)
        for i, team_ids in enumerate(player_ids):
            for j, player_id in enumerate(team_ids):
                last_n = self.get_last_n_games(game_date=game_date, 
                                                    poi_id=player_id, 
                                                    n=CONTEXT_WINDOW)[0]
                num_rows = last_n.shape[0]
                if num_rows < CONTEXT_WINDOW:
                    last_n = F.pad(
                        last_n,
                        (0, 0, 0, CONTEXT_WINDOW - num_rows)
                    )
                data[i][j] = last_n
        
        return poi_data, data

    
    def get_last_n_games(self, game_date: int, poi_id: int, n: int) -> Tuple[Tensor, DataFrame]:
        last_n_before_game = (
            DataAggregator.game_log[(DataAggregator.game_log["GAME_DATE"] <= game_date) 
                                    & (DataAggregator.game_log["PLAYER_ID"] == int(poi_id))]
            .sort_values("GAME_DATE", ascending=False)
            .head(n)
        )
        col_i = last_n_before_game.shape[1] - FEATURES
        data = tensor(last_n_before_game.iloc[:, col_i:].values, dtype=torch.float32)
        metadata = last_n_before_game.iloc[:, :col_i]
        return data, metadata