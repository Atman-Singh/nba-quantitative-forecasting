import numpy as np
from torch import tensor
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.datetime_helpers import DatetimeHelpers
from utils.params import CONTEXT_WINDOW
from .espn_client import ESPNClient

GAME_LOG_DIR = "data/game_logs"
MPG_TABLE_DIR = "data/mpg_tables/"
MPG_TABLE_PATH = MPG_TABLE_DIR + 'mpg_table.json'
CONFIG_PATH = os.path.join(GAME_LOG_DIR, "config.json")
PLAYER_IDS_PATH = "data/player_ids.json"

class RawDataAggregator():
    
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
    player_ids = pd.Series(dtype="int64")
    player_names = {}
    mpg_table = {}

    @staticmethod
    def _add_to_game_log(rows_df) -> None:
        RawDataAggregator.game_log = pd.concat([RawDataAggregator.game_log, rows_df], ignore_index=True)

    @staticmethod
    def build_game_log(years: int = 3, reload: bool = False, save_step: int = None) -> None:
        rows = []
        current_date = None
        if reload:
            current_date = date = datetime.strptime(
                RawDataAggregator.load_save_date(),
                "%Y-%m-%d %H:%M:%S.%f"
            )
        
        if current_date == None:
            current_date = date = DatetimeHelpers.get_current_date()
        else:
            RawDataAggregator.load_game_log()


        while (current_date - date).days <= years * 365:
            try:
                games = ESPNClient.get_scoreboard(date)['events']
            except KeyError:
                print('no games on ' + str(date))
            # TODO: parralelize fetching box scores for all games on a given date
            for game in games:
                game_id = int(game['id'])
                box_scores = ESPNClient.get_box_scores(game_id)
                if box_scores:
                    for team in box_scores:
                        for player in team['statistics'][0]['athletes']:
                            player_id = int(player['athlete'].get('id', -1))
                            if player_id == -1:
                                print('no id')
                                continue

                            ESPNClient.format_box_score(player['stats'])
                            row = [game_id, DatetimeHelpers._format_date(date), 1, player_id, int(not player['didNotPlay'])]
                            row.extend(player['stats'])
                            rows.append(row)
            date = DatetimeHelpers.decrement_date(date)

            if save_step and (current_date - date).days % save_step == 0:
                RawDataAggregator._add_to_game_log(pd.DataFrame(rows, columns=RawDataAggregator.game_log.columns))
                RawDataAggregator.save_game_log(str(date))
                rows = []
            
        RawDataAggregator._add_to_game_log(pd.DataFrame(rows, columns=RawDataAggregator.game_log.columns))
        RawDataAggregator.save_game_log()
    
    @staticmethod
    def save_game_log(save_date: str = None) -> None:
        name = "/game_log_" + DatetimeHelpers.get_timestamp() + '.parquet'
        path = GAME_LOG_DIR + name
        os.makedirs(GAME_LOG_DIR, exist_ok=True)
        RawDataAggregator.game_log.to_parquet(
            path=path,
            engine="pyarrow",
            index=False
        )

        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)

        config['game_log_name'] = name
        config['save_date'] = save_date if save_date else 'none'

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)

        print('Game log saved to ' + path)


    @staticmethod
    def load_save_name() -> str:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return config['game_log_name']

    @staticmethod
    def load_save_date() -> str | None:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        return None if config['save_date'] == 'none' else config['save_date']
    
    @staticmethod
    def load_game_log(columns: list = None) -> None:
        name = RawDataAggregator.load_save_name()

        if columns:
            RawDataAggregator.game_log = pd.read_parquet(GAME_LOG_DIR + name, 
                                   engine="pyarrow",
                                   columns=columns
                                   )
        else:
            RawDataAggregator.game_log = pd.read_parquet(GAME_LOG_DIR + name, engine="pyarrow")
        
        RawDataAggregator.load_player_ids()
        # RawDataAggregator.load_player_names()
    
    @staticmethod
    def load_player_names() -> None:
        if RawDataAggregator.player_ids is None:
            print('Load player ids.')
        else: 
            RawDataAggregator.player_names = {int(k): ESPNClient.get_player_name(int(k)) for k in RawDataAggregator.player_ids.keys()}
        print('Player names loaded.')
        
    @staticmethod
    def save_player_ids() -> None:
        os.makedirs("data", exist_ok=True)
        data = {str(pid): df.to_dict('records') for pid, df in RawDataAggregator.player_ids.items()}
        with open(PLAYER_IDS_PATH, 'w') as f:
            json.dump(data, f)
        print('Player IDs saved.')

    @staticmethod
    def load_player_ids() -> None:
        if os.path.exists(PLAYER_IDS_PATH):
            with open(PLAYER_IDS_PATH, 'r') as f:
                data = json.load(f)
            for pid_str, records in data.items():
                RawDataAggregator.player_ids[int(pid_str)] = pd.DataFrame(records)
            print('Player IDs loaded from cache.')
            return

        if RawDataAggregator.game_log is None or 'PLAYER_ID' not in RawDataAggregator.game_log:
            print('Load game log with the player ids column.')
        else:
            player_ids = RawDataAggregator.game_log['PLAYER_ID']
            for player_id in player_ids:
                dates = RawDataAggregator.game_log[RawDataAggregator.game_log['PLAYER_ID'] == player_id][['GAME_DATE', 'GAME_ID']].sort_values('GAME_DATE', ascending=False)
                print(player_id)
                RawDataAggregator.player_ids[player_id] = dates
            RawDataAggregator.save_player_ids()
        print('Player IDs loaded.')
    
    @staticmethod
    def get_players_game_on_date(player_id: int, date: datetime) -> pd.Series:
        if type(date) != int:
            date = DatetimeHelpers._format_date(date)
        return RawDataAggregator.game_log[(RawDataAggregator.game_log['GAME_DATE'] == date) 
                                          & (RawDataAggregator.game_log['PLAYER_ID'] == player_id)]
    
    @staticmethod
    def build_mpg_table() -> None:
        mpg_table = {}
        for player_id in RawDataAggregator.player_ids:
            print(ESPNClient.get_player_name(player_id))
            player_minutes = RawDataAggregator.game_log[RawDataAggregator.game_log['PLAYER_ID'] == player_id][['GAME_DATE', 'MIN']].sort_values("GAME_DATE", ascending=False)
            for date in player_minutes['GAME_DATE']:
                window = player_minutes[player_minutes['GAME_DATE'] <= date].head(CONTEXT_WINDOW)['MIN']
                key, value = int(player_id), float(window.mean())
                if key not in mpg_table:
                    mpg_table[key] = {
                        int(date): value
                    }
                else:
                    mpg_table[key][date] = value

        os.makedirs(MPG_TABLE_DIR, exist_ok=True)
        with open(MPG_TABLE_PATH, 'w') as f:
            json.dump(mpg_table, f)
    
    @staticmethod
    def load_mpg_table() -> None:
        os.makedirs(MPG_TABLE_DIR, exist_ok=True)
        with open(MPG_TABLE_PATH, 'r') as f:
            data = json.load(f)
            RawDataAggregator.mpg_table = {
                int(player_id): {int(date): float(value) for date, value in dates.items()}
                for player_id, dates in data.items()
            }

    @staticmethod
    def get_player_mpg(player_id: int, game_date: int, timezone_offset_days: int = 2) -> float:
        if player_id not in RawDataAggregator.mpg_table:
            raise KeyError(f"Player {player_id} not found in mpg_table")
        
        player_dates = RawDataAggregator.mpg_table[player_id]
        
        if game_date in player_dates:
            return player_dates[game_date]
        
        print(f"Checking nearby dates within {timezone_offset_days} days")

        game_date_dt = datetime.strptime(str(game_date), DatetimeHelpers.get_date_format())
        for offset in range(1, timezone_offset_days + 1):
            for offset_val in [-offset, offset]:
                offset_date_dt = game_date_dt + timedelta(days=offset_val)
                offset_date_int = int(offset_date_dt.strftime("%Y%m%d"))
                if offset_date_int in player_dates:
                    return player_dates[offset_date_int]
        
        print(f"No MPG found for player {player_id} on date {game_date} or nearby dates")
        try:
            key = RawDataAggregator.mpg_table[player_id].keys()[0]
        except:
            return None
        return RawDataAggregator.mpg_table[player_id][key]