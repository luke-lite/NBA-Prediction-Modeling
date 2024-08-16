import pandas as pd
import pickle

class ELOModelBuilder():
    def __init__(self, model_name, model_parameters, initial_elos, df):
        self.model_name = model_name
        # self.model_parameters = model_parameters # to-do: include K and win_prob parameters
        self.home_advantage = model_parameters['home_adv']
        self.initial_elos = initial_elos
        self.initial_df = df

    def __repr__(self):
        return f'ELO Model: {self.model_name})'
    
    def create_elo_history_dict(self):
        self.elo_dict = {}
        for team_id in self.initial_elos['team_id']:
            # empty lists for each team each season
            self.elo_dict[team_id] = {season: [] for season in self.initial_df['season'].unique()}
            # add starting elo into the first season for each team
            self.elo_dict[team_id][1314].append(self.initial_elos['elo_i'][self.initial_elos['team_id'] == team_id].iloc[0])


    def calc_K(self, MOV, elo_diff_winner):
        K = 20 * ( (MOV + 3)**0.8 / (7.5 + 0.006*(elo_diff_winner)) )
        return K
    
    def calc_win_probability(self, away_elo, home_elo):
        away_team_wp = 1/(1 + 10**(((home_elo+self.home_advantage)-away_elo)/400))
        home_team_wp = 1/(1 + 10**((away_elo-(home_elo+self.home_advantage))/400))

        E_away = away_team_wp
        E_home = home_team_wp

        return E_away, E_home
        
    def new_season_elo_adj(self, elo):
        new_season_elo = (0.75*elo) + (0.25*1505)
        return new_season_elo

    def update_elo(self, away_elo, away_score, home_elo, home_score):
        away_elo_og = away_elo
        home_elo_og = home_elo
        
        # set home court advantage
        home_adv = self.home_advantage
        home_elo += home_adv
        
        # determine winner/loser
        if away_score > home_score:
            winner_score = away_score
            winner_elo = away_elo
            
            loser_score = home_score
            loser_elo = home_elo
            
            S_away = 1
            S_home = 0
        else:
            winner_score = home_score
            winner_elo = home_elo
            
            loser_score = away_score
            loser_elo = away_elo
            S_away = 0
            S_home = 1

        E_away, E_home = self.calc_win_probability(away_elo, home_elo)
        
        MOV = winner_score - loser_score
        elo_diff_winner = winner_elo - loser_elo
        K = self.calc_K(MOV=MOV, elo_diff_winner=elo_diff_winner)
        
        # calculate new elo
        away_elo_new = K*(S_away-E_away) + away_elo_og
        home_elo_new = K*(S_home-E_home) + home_elo_og
        
        return away_elo_new, home_elo_new
    
    def create_new_model(self):
        # add initial and updated elo s to each game row
        self.initial_df[['away_elo_i', 'away_elo_n', 'home_elo_i', 'home_elo_n']] = None
        seasons = list(self.initial_df['season'].unique())

        for idx in self.initial_df.index:
            # store current season to determine when to perform season rating adjustment
            curr_season = self.initial_df.loc[idx]['season']
            # when current season list is empty, trigger season rating adjustment for away team
            if self.elo_dict[self.initial_df.loc[idx,'away_team']][curr_season] == []:
                
                prev_season_idx = seasons.index(curr_season)-1
                prev_elo = self.elo_dict[self.initial_df.loc[idx,'away_team']][seasons[prev_season_idx]][-1]
                new_season_elo = self.new_season_elo_adj(prev_elo)
                self.elo_dict[self.initial_df.loc[idx,'away_team']][curr_season].append(new_season_elo)
            # when current season list is empty, trigger season rating adjustment for home team
            if self.elo_dict[self.initial_df.loc[idx,'home_team']][curr_season] == []:
                
                prev_season_idx = seasons.index(curr_season)-1
                prev_elo = self.elo_dict[self.initial_df.loc[idx,'home_team']][seasons[prev_season_idx]][-1]
                new_season_elo = self.new_season_elo_adj(prev_elo)
                self.elo_dict[self.initial_df.loc[idx,'home_team']][curr_season].append(new_season_elo)
                
            # determine current elo before each game
            away_elo_current = self.elo_dict[self.initial_df.loc[idx,'away_team']][curr_season][-1]
            home_elo_current = self.elo_dict[self.initial_df.loc[idx,'home_team']][curr_season][-1]
            
            # update elo after each game
            away_elo_new, home_elo_new = self.update_elo(away_team = self.initial_df.loc[idx,'away_team'],
                                                         away_elo = away_elo_current,
                                                         away_score = self.initial_df.loc[idx,'away_score'],
                                                         home_team = self.initial_df.loc[idx,'home_team'],
                                                         home_elo = home_elo_current,
                                                         home_score = self.initial_df.loc[idx,'home_score'])
            # store elos in game_info
            self.initial_df['away_elo_i'][idx] = away_elo_current
            self.initial_df['away_elo_n'][idx] = away_elo_new
            self.initial_df['home_elo_i'][idx] = home_elo_current
            self.initial_df['home_elo_n'][idx] = home_elo_new
            
            # store elos in elo_dict
            self.elo_dict[self.initial_df.loc[idx,'away_team']][curr_season].append(away_elo_new)
            self.elo_dict[self.initial_df.loc[idx,'home_team']][curr_season].append(home_elo_new)

    
    def save_elo_dict(self):
        elo_pkl_file = f"{self.model_name}_elo_dict.pkl"  

        with open(elo_pkl_file, 'wb') as file:  
            pickle.dump(self.elo_dict, file)

    def run(self):
        self.create_elo_history_dict()
        self.create_new_model()