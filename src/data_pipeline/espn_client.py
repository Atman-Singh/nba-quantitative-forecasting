import requests
import datetime as dt
from datetime import datetime
from datetime import date
import torch
from torch import tensor
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.datetime_helpers import DatetimeHelpers

FEATURES = 14
TEAMS_URL = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
SCOREBOARD_URL = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
BOX_SCORES_URL = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event='
ATHLETES_URL = 'https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/athletes/'
TEAMS = requests.get(TEAMS_URL).json()['sports'][0]['leagues'][0]['teams']
CURRENT_YEAR = date.today().year


# TODO: cache API responses

class ESPNClient: 

    @staticmethod
    def get_scoreboard(date: datetime) -> dict:
        params = {'dates':str(DatetimeHelpers._format_date(date))}
        return requests.get(SCOREBOARD_URL, params=params).json()
    
    @staticmethod
    def _format_team_name(team_name: str):
        return team_name[0].upper() + team_name[1:]
    
    @staticmethod
    def get_team_id(team_name: str):
        team_name = ESPNClient._format_team_name(team_name)
        for team in TEAMS:
            if team['team']['name'] == team_name:
                return team['team']['id']
    
    @staticmethod
    def get_team_name(team_id: int):
        return TEAMS[team_id-1]['team']['name']
    
    @staticmethod
    def get_schedule(team_id: int, season: int, season_type: int) -> list:
        if season_type > 3:
            Exception(f'season_type {season_type} invalid')

        url = f'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/schedule'

        if season_type != -1:
            # particular season_type
            params = {
                "season": season,
                "seasontype": season_type,
            }
            schedule = requests.get(url, params=params).json()['events']
        else:
            # all season types
            schedule = []
            for i in range(1, 4):
                params = {
                    "season": season,
                    "seasontype": i,
                }
                schedule.extend(requests.get(url, params=params).json()['events'])
        schedule.sort(key=lambda x: x["date"], reverse=True)
        return schedule  
    
    @staticmethod
    def get_last_n_matchups(team_a_id: int, team_b_id: int, n: int) -> list:
        team_b_name = ESPNClient.get_team_name(team_b_id)
        matchups = []
        i, m = 0, 0
        while m < n:
            schedule = ESPNClient.get_schedule(team_a_id, CURRENT_YEAR-i, -1)
            matchups.extend([e.get('id') for e in schedule if team_b_name in e.get("name")])
            m = len(matchups)
            if m < n:
                print(f'more games ({n}) requested than available, fetching more games')
            i += 1
        return matchups[:min(m, n)]

    @staticmethod
    def _get_last_n_game_ids(team_id: int, n: int) -> list:
        schedule = ESPNClient.get_schedule(team_id, CURRENT_YEAR, -1)
        ids = [r['id'] for r in schedule]
        if len(schedule) < n:
            print(f'warning: more games ({n}) requested than available')
        return ids[:min(len(schedule), n)]
    
    @staticmethod
    def format_box_score(box_score: list) -> list:
        for i, stat in enumerate(box_score):
            try:
                box_score[i] = int(stat)
            except:
                try:
                    if stat == '--':
                        box_score[i] = 0
                    else:
                        div = stat.split('-')
                        if int(div[1]) == 0:
                            box_score[i] = 0
                        else:
                            box_score[i] = int(div[0]) / int(div[1])
                except:
                    box_score[i] = 0

    @staticmethod
    def _get_player_box_scores(game_ids: list, player_id: str, team_id: int) -> tensor:
        k = len(game_ids)
        player_box_scores = torch.empty(k, FEATURES)
        for i, id in enumerate(game_ids):
            url = BOX_SCORES_URL + id
            teams = requests.get(url=url).json()['boxscore']['players']
            box_scores = []
            for team in teams:
                if team['team']['id'] == str(team_id):
                    box_scores = team['statistics'][0]['athletes']
            if len(box_scores) == 0:
                print('box scores not found')
                break
            
            for entry in box_scores:
                if entry['athlete']['id'] == player_id:
                    box_score = ESPNClient.format_box_score(entry['stats'])
                    player_box_scores[i] = tensor(box_score)
                    continue
            
        return player_box_scores
    
    @staticmethod
    def get_box_scores(game_id: int):
        url = BOX_SCORES_URL + str(game_id)
        try:
            return requests.get(url=url).json()['boxscore']['players']
        except KeyError:
            print('Game is yet to be played.')
         
    # get ids of teammate and opponents relative to a particular player id from a game id
    @staticmethod
    def get_player_ids(game_id: int, player_id: str) -> tuple[list[int], list[int]]:
        teammates = []
        opponents = []
        box_scores = ESPNClient.get_box_scores(game_id=game_id)
        
        poi_team_id = -1
        temp = []
        for team in box_scores:
            num_athletes = len(team['statistics'][0]['athletes'])
            cur_team_id = team['team']['id']
            if poi_team_id == -2:
                poi_team_id = cur_team_id
            for i, entry in enumerate(team['statistics'][0]['athletes']):
                try:
                    cur_id = entry['athlete']['id']
                except KeyError:
                    print('No ID')
                    continue
                if poi_team_id == -1:
                    if cur_id == player_id:
                        teammates = temp
                        poi_team_id = cur_team_id
                    elif i == num_athletes-1:
                        temp.append(int(cur_id))
                        opponents = temp
                        poi_team_id = -2
                    else:
                        temp.append(int(cur_id))
                elif cur_id != player_id:
                    if cur_team_id == poi_team_id:
                        teammates.append(int(cur_id))
                    else:
                        opponents.append(int(cur_id))
        
        return teammates, opponents
    
    @staticmethod
    def get_game_date(game_id: int) -> int:
        url = BOX_SCORES_URL + str(game_id)
        summary = requests.get(url=url).json()["header"]["competitions"][0]["date"]
        dt = datetime.fromisoformat(
            summary.replace("Z", "+00:00")
        )
        
        return DatetimeHelpers._format_date(dt)
    
    @staticmethod
    def get_player_name(player_id: int) -> str:
        url = ATHLETES_URL + str(player_id)
        return requests.get(url=url).json()['displayName']