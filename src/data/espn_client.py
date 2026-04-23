import requests
from datetime import date
import numpy as np

class ESPNClient: 
    def __init__(self):
        teams_url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams'
        
        self.box_scores_url = 'http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event='
        self.teams = requests.get(teams_url).json()['sports'][0]['leagues'][0]['teams']
        self.current_year = date.today().year
    
    def _format_team_name(self, team_name: str):
        return team_name[0].upper() + team_name[1:]
    
    def get_team_id(self, team_name: str):
        team_name = self._format_team_name(team_name)
        for team in self.teams:
            if team['team']['name'] == team_name:
                return team['team']['id']
    
    def get_team_name(self, team_id: int):
        return self.teams[team_id-1]['team']['name']
    
    def get_schedule(self, team_id: int, season: int, season_type: int) -> list:
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
    
    def get_last_n_games(self, team_id: int, n: int) -> list:
        schedule = self.get_schedule(team_id, self.current_year, -1)
        ids = [e['id'] for e in schedule]
        if len(schedule) < n:
            print(f'warning: more games ({n}) requested than available')
        return ids[:min(len(schedule), n)]
    
    def get_last_n_matchups(self, team_a_id: int, team_b_id: int, n: int) -> list:
        team_b_name = self.get_team_name(team_b_id)
        matchups = []
        i, m = 0, 0
        while m < n:
            schedule = self.get_schedule(team_a_id, self.current_year-i, -1)
            matchups.extend([e.get('id') for e in schedule if team_b_name in e.get("name")])
            m = len(matchups)
            if m < n:
                print(f'more games ({n}) requested than available, fetching more games')
            i += 1
        last_n = matchups[:min(m, n)]
        box_scores = self._get_box_scores(last_n)
        return last_n

    def _get_box_scores(self, match_ids: list):
        for id in match_ids:
            url = self.box_scores_url + id
            box_scores = requests.get(url=url).json()['boxscore']['players'][0]['statistics'][0]['athletes']
            print(len(box_scores))



