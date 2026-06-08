import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from pandas import DataFrame
from typing import Tuple
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.datetime_helpers import DatetimeHelpers
from utils.params import MAX_ROSTER_SIZE, CONTEXT_WINDOW, FEATURES
from .raw_data_aggregator import RawDataAggregator
from .espn_client import ESPNClient

GAMES_PER_YEAR = 60
DATASET_DIR = r"data/datasets"
DATASET_CONFIG_PATH = os.path.join(DATASET_DIR, "config.json")

class DatasetBuilder:
    
    def __init__(self):
        RawDataAggregator.load_game_log()
        self.total = 0
        self.cache_accesses = 0
        self.cache = {}
        self.dataset = None

    # build x years worth of overlapping sequences for every player in the league
    def build_dataset(self, years: int = 3, reload: bool = False, save_step: int = None) -> None:
        num_players = len(RawDataAggregator.player_ids)
        max_games = GAMES_PER_YEAR * years

        start_i = 0
        if reload:
            start_i = self.load_save_index()
            self.load_dataset()
            dataset = self.dataset
        else:
            dataset = torch.zeros((num_players,
                                max_games,
                                3,
                                MAX_ROSTER_SIZE,
                                CONTEXT_WINDOW,
                                FEATURES),
                                dtype=torch.float32)
        
        for i, player_id in enumerate(RawDataAggregator.player_ids):
            if i < start_i:
                continue

            current_date = date = DatetimeHelpers.get_current_date()
            game, offset = None, -1
            while game is None or game.empty:
                if (current_date - date).days > years * 365:
                    break
                game = RawDataAggregator.get_players_game_on_date(player_id=player_id,
                                                              date=date)
                date = DatetimeHelpers.decrement_date(date)
                offset += 1

            if game is None or game.empty:
                continue

            j = 0
            while j < max_games and ((current_date - date).days <= years * 365):
                if (game.empty):
                    print('empty')
                if game is not None and not game.empty:
                    game_id = game["GAME_ID"].values[0]
                    dataset[i][j] = self.build_player_trends(game_id, player_id, dataset[i][j])
                    j += 1

                date = DatetimeHelpers.decrement_date(date)
                game = RawDataAggregator.get_players_game_on_date(player_id=player_id,
                                                              date=date)

            if save_step and (i + 1) % save_step == 0:
                self.dataset = dataset
                self.save_dataset(save_index=i + 1)
                print(f"Checkpoint saved after player index {i + 1}")
                
        self.dataset = dataset
        self.save_dataset()
        
    def save_dataset(self, save_index: int = None) -> None:
        os.makedirs(DATASET_DIR, exist_ok=True)
        for f in Path(DATASET_DIR).glob("*.pt"):
            f.unlink()
        name = "/dataset_" + DatetimeHelpers.get_timestamp() + '.pt'
        dataset_path = DATASET_DIR + name
        torch.save(self.dataset, dataset_path)
        print(f"Dataset saved at {dataset_path}")

        config = {}
        if os.path.exists(DATASET_CONFIG_PATH):
            with open(DATASET_CONFIG_PATH, 'r') as f:
                config = json.load(f)

        config['dataset_name'] = name
        config['save_index'] = save_index if save_index is not None else 'none'

        with open(DATASET_CONFIG_PATH, 'w') as f:
            json.dump(config, f)

    def load_save_index(self) -> int:
        with open(DATASET_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return 0 if config['save_index'] == 'none' else config['save_index']

    def load_dataset(self) -> None:
        with open(DATASET_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        name = config['dataset_name']
        self.dataset = torch.load(DATASET_DIR + name, weights_only=True)
            
        

    def build_player_trends(self, game_id: int, poi_id: int, data: Tensor) -> Tensor:
        game_date = ESPNClient.get_game_date(game_id)
        print(data.shape)
        last_5 = self.get_last_n_games(game_date, poi_id, CONTEXT_WINDOW)[0]
        print(last_5.shape)
        data[0][0] = last_5

        player_ids = ESPNClient.get_player_ids(game_id, poi_id)
        for i, team_ids in enumerate(player_ids, start=1):
            RawDataAggregator.load_mpg_table()
            top_k_ids = DataFrame(columns=["PLAYER_IDS", "MPG"])
            top_k_ids["PLAYER_IDS"] = team_ids
            top_k_ids["MPG"] = [RawDataAggregator.get_player_mpg(i, game_date) for i in top_k_ids["PLAYER_IDS"]]
            top_k_ids = top_k_ids.sort_values("MPG", ascending=False).head(MAX_ROSTER_SIZE)

            for j, player_id in enumerate(top_k_ids['PLAYER_IDS']):
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
        return data

    
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

        num_rows = data.shape[0]
        if num_rows < n:
            data = F.pad(
                data,
                (0, 0, 0, n - num_rows)
            )
        return data, metadata
