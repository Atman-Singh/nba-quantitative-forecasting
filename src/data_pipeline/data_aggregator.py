from .espn_client import ESPNClient
import numpy as np
from torch import tensor
import pandas as pd
from datetime import datetime
import os
import json

GAME_LOG_DIR = "data/game_logs"
TIMESTAMP_FORMAT = r"%Y%m%d%H%M%S"
CONFIG_PATH = os.path.join(GAME_LOG_DIR, "config.json")

class DataAggregator():
    
    game_log = pd.DataFrame({
        "GAME_ID": pd.Series(dtype="int64"),
        "GAME_DATE": pd.Series(dtype="int64"),
        "REAL_PLAYER": pd.Series(dtype="int64"),
        "PLAYER_ID": pd.Series(dtype="int64"),
        "PLAYING": pd.Series(dtype="int64"),
        "MIN": pd.Series(dtype="int64"),
        "PTS": pd.Series(dtype="float64"),
        "FG": pd.Series(dtype="float64"),
        "3PT": pd.Series(dtype="float64"),
        "FT": pd.Series(dtype="float64"),
        "REB": pd.Series(dtype="int64"),
        "AST": pd.Series(dtype="int64"),
        "TO": pd.Series(dtype="int64"),
        "STL": pd.Series(dtype="int64"),
        "BLK": pd.Series(dtype="int64"),
        "OREB": pd.Series(dtype="int64"),
        "DREB": pd.Series(dtype="int64"),
        "PF": pd.Series(dtype="int64"),
        "PM": pd.Series(dtype="int64"),
    })

    @staticmethod
    def _add_to_game_log(rows_df) -> None:
        DataAggregator.game_log = pd.concat([DataAggregator.game_log, rows_df], ignore_index=True)

    @staticmethod
    def build_player_game_logs() -> None:
        rows = []
        date = ESPNClient.get_current_date()
        for _ in range(50):
            games = ESPNClient.get_scoreboard(date)['events']
            for game in games:
                game_id = int(game['id'])
                box_scores = ESPNClient.get_box_scores(game_id)
                if box_scores:
                    for team in box_scores:
                        for player in team['statistics'][0]['athletes']:
                            player_id = int(player['athlete']['id'])
                            ESPNClient.format_box_score(player['stats'])
                            row = [game_id, ESPNClient._format_date(date), 1, player_id, int(not player['didNotPlay'])]
                            row.extend(player['stats'])
                            rows.append(row)
            date = ESPNClient.decrement_date(date)
        DataAggregator._add_to_game_log(pd.DataFrame(rows, columns=DataAggregator.game_log.columns))
        DataAggregator.save_game_log()
    
    @staticmethod
    def save_game_log() -> None:
        name = "/game_log_" + datetime.now().strftime(TIMESTAMP_FORMAT) + '.parquet'
        path = GAME_LOG_DIR + name
        os.makedirs(GAME_LOG_DIR, exist_ok=True)
        DataAggregator.game_log.to_parquet(
            path=path,
            engine="pyarrow",
            index=False
        )

        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
        config['game_log_name'] = name
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)

        print('Game log saved to ' + path)
    
    @staticmethod
    def load_game_log(columns: list = None) -> None:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        name = config['game_log_name']

        if columns:
            DataAggregator.game_log = pd.read_parquet(GAME_LOG_DIR + name, 
                                   engine="pyarrow",
                                   columns=columns
                                   )
        DataAggregator.game_log = pd.read_parquet(GAME_LOG_DIR + name, engine="pyarrow")
        