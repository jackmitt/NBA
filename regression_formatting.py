import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import statsmodels.api as sm
#import arviz as az
#import pymc as pm
#import nutpie
#import pytensor
import pickle
import math
import matplotlib.pyplot as plt
import time
import concurrent.futures
from scipy.optimize import minimize

def first():
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    p_priors = {'3pt_pct_off':{},'2pt_pct_off':{},'ft_pct':{},'to_pct_off':{},"ftr_off":{}}
    t_priors = {'3pt_pct_def':{},'2pt_pct_def':{},'to_pct_def':{},"ftr_def":{},'oreb_pct':{},'dreb_pct':{}}
    for i in (players):
        for key in p_priors:
            p_priors[key][i] = [0,0]
    for i in teams:
        for key in t_priors:
            t_priors[key][i] = [0,0]
    
    #use arma method for these
    for i in teams:
        t_priors['oreb_pct'][i] = 0.3
        t_priors['dreb_pct'][i] = 0.7


    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}



    #Using optimal values determined in other_latent_models.py
    fatten = {'3pt_pct_off':0.997,'2pt_pct_off':0.99,'ft_pct':0.98,'to_pct_off':0.98,"ftr_off":0.97,'3pt_pct_def':(0.995,1),'2pt_pct_def':(0.974,0.5),
              'to_pct_def':(0.96,0.8),"ftr_def":(0.96,0.7),'oreb_pct':0.05,'dreb_pct':0.05}


    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec

            


            for x in [h_id, a_id]:
                cur_f = {}
                for y in ['last_','cur_']:
                    for key in p_priors:
                        cur_f[y+'pred_'+key] = 0

                if (x == h_id):
                    cur_f['home'] = 1
                else:
                    cur_f['home'] = 0
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0] > 0):
                        cur_f['last_pred_3pt_pct_off'] += last_lineup[x][row['player_id']] * (p_priors['3pt_pct_off'][row['player_id']][0] / (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0]))
                    
                    
                    
                    if (cur_season != "1996-97"):
                        features.append(cur_f)

                    #Update
                    p_priors['3pt_pct_off'][row['player_id']][0] += row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][1] += row['threePointersAttempted'] - row['threePointersMade']

                    p_priors['3pt_pct_off'][row['player_id']][0] *= fatten['3pt_pct_off']
                    p_priors['3pt_pct_off'][row['player_id']][1] *= fatten['3pt_pct_off']


            for x in [h_id, a_id]:
                last_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    last_lineup[x][row['player_id']] = row['seconds']/total_sec

    
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_3pt_pct_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))