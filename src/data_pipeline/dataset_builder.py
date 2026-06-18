import torch
from torch import tensor, Tensor
import torch.nn.functional as F
from pandas import DataFrame
from typing import Tuple
import os
import json
import time
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
    
    def __init__(self, timing: bool = False):
        self.timing = timing
        RawDataAggregator.load_game_log()
        self._timed("load_mpg_table", RawDataAggregator.load_mpg_table)
        self.dataset = None

        self.cache = {}
        self.player_id_cache = {}

    def _timed(self, label: str, fn, *args, **kwargs):
        if not self.timing:
            return fn(*args, **kwargs)
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        print(f"  [{label}] {time.perf_counter() - start:.4f}s")
        return result

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
        
        if RawDataAggregator.player_ids is None:
            print('No player IDs')
            return
        
        for i, player_id in enumerate(RawDataAggregator.player_ids.keys()):
            print(player_id)
            if i < start_i:
                continue

            dates = RawDataAggregator.player_ids[player_id]

            if dates is None:
                print('No dates.')
                continue

            player_start = time.perf_counter() if self.timing else None
            j = 0
            for date, game_id in dates.values:
                if j >= max_games:
                    break
                game_start = time.perf_counter() if self.timing else None
                dataset[i][j] = self.build_player_trends(game_id, player_id, dataset[i][j], date)
                if self.timing:
                    print(f"  [game {j}] {time.perf_counter() - game_start:.4f}s")
                j += 1
            if self.timing:
                print(f"[player {i+1} ({player_id})] {time.perf_counter() - player_start:.4f}s total")

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
            
        

    def build_player_trends(self, game_id: int, poi_id: int, data: Tensor, game_date: int = None) -> Tensor:
        print(game_date)
        if game_date is None:
            game_date = self._timed("get_game_date", ESPNClient.get_game_date, game_id)
        last_5 = self._timed("get_last_n_games", self.get_last_n_games, game_date, poi_id, CONTEXT_WINDOW)[0]
        data[0][0] = last_5

        if game_id in self.player_id_cache:
            print('PI cache accessed')
            all_player_ids = self.player_id_cache[game_id]
        else:
            all_player_ids = self.player_id_cache[game_id] = self._timed("get_player_ids", ESPNClient.get_player_ids, game_id)
        poi_team_id = self._timed("get_player_team_id", RawDataAggregator.get_player_team_id, poi_id, game_id)
        player_ids = self._timed("get_teammate_and_opponent_ids", ESPNClient.get_teammate_and_opponent_ids, all_player_ids, poi_team_id)

        for i, team_ids in enumerate(player_ids, start=1):
            top_k_ids = DataFrame(columns=["PLAYER_IDS", "MPG"])
            top_k_ids["PLAYER_IDS"] = team_ids
            top_k_ids["MPG"] = self._timed(
                "get_player_mpg",
                lambda: [RawDataAggregator.get_player_mpg(pid, game_date) for pid in top_k_ids["PLAYER_IDS"]]
            )
            top_k_ids = top_k_ids.sort_values("MPG", ascending=False).head(MAX_ROSTER_SIZE)

            for j, player_id in enumerate(top_k_ids['PLAYER_IDS']):
                if player_id == poi_id:
                    continue

                key = (player_id, game_date)
                if key in self.cache:
                    print('Cache accessed')
                    data[i][j] = self.cache[key]
                    continue  
                      
                last_n = self._timed(
                    "get_last_n_games (teammate)",
                    self.get_last_n_games,
                    game_date=game_date,
                    poi_id=player_id,
                    n=CONTEXT_WINDOW
                )[0]
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
