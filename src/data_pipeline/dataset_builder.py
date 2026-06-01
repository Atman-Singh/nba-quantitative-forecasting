import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from pandas import DataFrame
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.params import CONTEXT_WINDOW
from .raw_data_aggregator import RawDataAggregator
from .espn_client import ESPNClient

FEATURES = 17
MAX_ROSTER_SIZE = 8

class DatasetBuilder:
    
    def __init__(self):
        RawDataAggregator.load_game_log()
        self.total = 0
        self.cache_accesses = 0
        self.cache = {}

    # build x years worth of overlapping sequences for every player in the league
    def build_dataset(self, years: int = 3): 
        for player_id in RawDataAggregator.player_ids:
            current_date = date = ESPNClient.get_current_date()
            game, offset = None, -1
            while game is None or game.empty:
                game = RawDataAggregator.get_players_game_on_date(player_id=player_id,
                                                              date=date)
                date = ESPNClient.decrement_date(date)
                offset += 1
                
            while (current_date - date).days <= years * 365 + offset:
                if (game.empty):
                    print('empty')
                print(player_id, date)
                if game is not None and not game.empty:
                    game_id = game["GAME_ID"].values[0]
                    trends = self.build_player_trends(game_id, player_id)

                date = ESPNClient.decrement_date(date)
                game = RawDataAggregator.get_players_game_on_date(player_id=player_id,
                                                              date=date)
    
    def build_player_trends(self, game_id: int, poi_id: int) -> Tensor:
        game_date = ESPNClient.get_game_date(game_id)
        poi_data = self.get_last_n_games(game_date, poi_id, CONTEXT_WINDOW)[0]

        player_ids = ESPNClient.get_player_ids(game_id, poi_id)
        data = torch.empty((2, MAX_ROSTER_SIZE, CONTEXT_WINDOW, FEATURES), 
                            dtype=torch.float32)
        for i, team_ids in enumerate(player_ids):
            for j, player_id in enumerate(team_ids):
                self.total += 1
                key = (player_id, game_date)
                if key in self.cache:
                    print(f'Accessed cache, cache len {len(self.cache)}')
                    self.cache_accesses += 1
                    data[i][j] = self.cache[key]
                    continue  
                      
                last_n = self.get_last_n_games(game_date=game_date, 
                                                    poi_id=player_id, 
                                                    n=CONTEXT_WINDOW)[0]
                num_rows = last_n.shape[0]
                if num_rows < CONTEXT_WINDOW:
                    last_n = F.pad(
                        last_n,
                        (0, 0, 0, CONTEXT_WINDOW - num_rows)
                    )
                data[i][j] = self.cache[key] = last_n
        
        return poi_data, data

    
    def get_last_n_games(self, game_date: int, poi_id: int, n: int) -> Tuple[Tensor, DataFrame]:
        last_n_before_game = (
            RawDataAggregator.game_log[(RawDataAggregator.game_log["GAME_DATE"] <= game_date) 
                                    & (RawDataAggregator.game_log["PLAYER_ID"] == int(poi_id))]
            .sort_values("GAME_DATE", ascending=False)
            .head(n)
        )
        col_i = last_n_before_game.shape[1] - FEATURES
        data = tensor(last_n_before_game.iloc[:, col_i:].values, dtype=torch.float32)
        metadata = last_n_before_game.iloc[:, :col_i]
        return data, metadata
