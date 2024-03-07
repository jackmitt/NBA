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
    team_adv_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    for i in range(len(player_bs.index)):
        if (not pd.isnull(player_bs.at[i,'minutes']) and ':' not in player_bs.at[i,'minutes']):
            player_bs.at[i,'minutes'] = player_bs.at[i,'minutes'] + ':00:00'

            
    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

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

    last_season = -1
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            team_adv_game = team_adv_bs.loc[team_adv_bs["game_id"] == gid,].reset_index()
            
            if (last_season != -1 and cur_season != last_season):
                for team in teams:
                    for j in range(2):
                        t_priors['3pt_pct_def'][team][j] *= fatten['3pt_pct_def'][1]
                        t_priors['2pt_pct_def'][team][j] *= fatten['2pt_pct_def'][1]
                        t_priors['to_pct_def'][team][j] *= fatten['to_pct_def'][1]
                        t_priors['ftr_def'][team][j] *= fatten['ftr_def'][1]
            last_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
                h_eff = team_adv_game.at[0,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec

            


            for x in [h_id, a_id]:
                if (x == h_id):
                    y = a_id
                else:
                    y = h_id
                
                cur_f = {'game_id':team_game.at[0,'game_id'],'date':date,'season':cur_season,'off_team_id':x,'def_team_id':y}

                for yyy in ['last_','cur_']:
                    for key in p_priors:
                        if (key not in ['ftr_off','ft_pct']):
                            cur_f[yyy+'pred_'+key] = 0
                    cur_f[yyy+'pred_ftrXftp_off'] = 0
                    cur_f[yyy+'pred_ftrXftp_def'] = 0

                if (x == h_id):
                    cur_f['home_off'] = 1
                else:
                    cur_f['home_off'] = 0


                for player in last_lineup[x]:
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0] > 0):
                        cur_f['last_pred_3pt_pct_off'] += last_lineup[x][player] * (p_priors['3pt_pct_off'][player][0] / (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0]))
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0] > 0):
                        cur_f['last_pred_2pt_pct_off'] += last_lineup[x][player] * (p_priors['2pt_pct_off'][player][0] / (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0]))
                    #to_pct_off
                    if (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0] > 0):
                        cur_f['last_pred_to_pct_off'] += last_lineup[x][player] * (p_priors['to_pct_off'][player][0] / (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0]))
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0] > 0):
                        cur_f['last_pred_ftrXftp_off'] += last_lineup[x][player] * (p_priors['ftr_off'][player][0] / (p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        cur_f['last_pred_ftrXftp_def'] += last_lineup[x][player] * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))

                #player features
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_3pt_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['3pt_pct_off'][row['player_id']][0] / (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0]))
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_2pt_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['2pt_pct_off'][row['player_id']][0] / (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0]))
                    #to_pct_off
                    if (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_to_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['to_pct_off'][row['player_id']][0] / (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0]))
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_ftrXftp_off'] += cur_lineup[x][row['player_id']] * (p_priors['ftr_off'][row['player_id']][0] / (p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        cur_f['cur_pred_ftrXftp_def'] += cur_lineup[x][row['player_id']] * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))


                    #Update player stuff
                    p_priors['3pt_pct_off'][row['player_id']][0] += row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][1] += row['threePointersAttempted'] - row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][0] *= fatten['3pt_pct_off']
                    p_priors['3pt_pct_off'][row['player_id']][1] *= fatten['3pt_pct_off']

                    p_priors['2pt_pct_off'][row['player_id']][0] += (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][1] += (row['fieldGoalsAttempted'] - row['threePointersAttempted']) - (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][0] *= fatten['2pt_pct_off']
                    p_priors['2pt_pct_off'][row['player_id']][1] *= fatten['2pt_pct_off']

                    p_priors['ft_pct'][row['player_id']][0] += row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][1] += row['freeThrowsAttempted'] - row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][0] *= fatten['ft_pct']
                    p_priors['ft_pct'][row['player_id']][1] *= fatten['ft_pct']

                    p_priors['to_pct_off'][row['player_id']][0] += row['turnovers']
                    p_priors['to_pct_off'][row['player_id']][1] += row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists']
                    p_priors['to_pct_off'][row['player_id']][0] *= fatten['to_pct_off']
                    p_priors['to_pct_off'][row['player_id']][1] *= fatten['to_pct_off']

                    p_priors['ftr_off'][row['player_id']][0] += row['freeThrowsAttempted']
                    p_priors['ftr_off'][row['player_id']][1] += row['fieldGoalsAttempted']
                    p_priors['ftr_off'][row['player_id']][0] *= fatten['ftr_off']
                    p_priors['ftr_off'][row['player_id']][1] *= fatten['ftr_off']

                #team features
                if (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0] > 0):
                    cur_f['pred_3pt_pct_def'] = (t_priors['3pt_pct_def'][y][0] / (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0]))
                if (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0] > 0):
                    cur_f['pred_2pt_pct_def'] = (t_priors['2pt_pct_def'][y][0] / (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0]))
                if (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0] > 0):
                    cur_f['pred_to_pct_def'] = (t_priors['to_pct_def'][y][0] / (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0]))
                cur_f['pred_oreb_pct_off'] = t_priors['oreb_pct'][x]
                cur_f['pred_oreb_pct_def'] = 1 - t_priors['dreb_pct'][y]

                if (x == h_id):
                    ind = 0
                    rev_ind = 1
                else:
                    ind = 1
                    rev_ind = 0
                
                #Update
                t_priors['3pt_pct_def'][y][0] += team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][1] += team_game.at[ind,'threePointersAttempted'] - team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][0] *= fatten['3pt_pct_def'][0]
                t_priors['3pt_pct_def'][y][1] *= fatten['3pt_pct_def'][0]

                t_priors['2pt_pct_def'][y][0] += team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade']
                t_priors['2pt_pct_def'][y][1] += (team_game.at[ind,'fieldGoalsAttempted'] - team_game.at[ind,'threePointersAttempted']) - (team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade'])
                t_priors['2pt_pct_def'][y][0] *= fatten['2pt_pct_def'][0]
                t_priors['2pt_pct_def'][y][1] *= fatten['2pt_pct_def'][0]

                t_priors['to_pct_def'][y][0] += team_game.at[ind,'turnovers']
                t_priors['to_pct_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted'] + team_game.at[ind,'freeThrowsAttempted']*0.44
                t_priors['to_pct_def'][y][0] *= fatten['to_pct_def'][0]
                t_priors['to_pct_def'][y][1] *= fatten['to_pct_def'][0]

                t_priors['ftr_def'][y][0] += team_game.at[ind,'freeThrowsAttempted']
                t_priors['ftr_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted']
                t_priors['ftr_def'][y][0] *= fatten['ftr_def'][0]
                t_priors['ftr_def'][y][1] *= fatten['ftr_def'][0]

                pred_oreb_pct = (t_priors['oreb_pct'][x] + (1 - t_priors['dreb_pct'][y])) / 2
                actual_oreb_pct = team_game.at[ind,'reboundsOffensive'] / (team_game.at[ind,'reboundsOffensive'] + team_game.at[rev_ind,'reboundsDefensive'])
                t_priors['oreb_pct'][x] += (actual_oreb_pct - pred_oreb_pct) * fatten['oreb_pct']
                t_priors['dreb_pct'][y] += ((1-actual_oreb_pct) - (1-pred_oreb_pct)) * fatten['dreb_pct']
                
                cur_f['actual_eff'] = team_adv_game.at[ind,'offensiveRating']

                if (cur_season != "1996-97"):
                    features.append(cur_f)

            for x in [h_id, a_id]:
                last_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    last_lineup[x][row['player_id']] = row['seconds']/total_sec

    
    fdf = pd.DataFrame(features)

    fdf.to_csv("./intermediates/regression_formatted/first.csv", index=False)

#first but all the player based stats are weighted according to each players usage and playtime instead of just playtime
def second():
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    team_adv_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    usg_bs = pd.read_csv("./database/usage_boxscores_players.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    for i in range(len(player_bs.index)):
        if (not pd.isnull(player_bs.at[i,'minutes']) and ':' not in player_bs.at[i,'minutes']):
            player_bs.at[i,'minutes'] = player_bs.at[i,'minutes'] + ':00:00'

            
    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
    usg_bs['seconds'] = usg_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    p_priors = {'3pt_pct_off':{},'2pt_pct_off':{},'ft_pct':{},'to_pct_off':{},"ftr_off":{}}
    t_priors = {'3pt_pct_def':{},'2pt_pct_def':{},'to_pct_def':{},"ftr_def":{},'oreb_pct':{},'dreb_pct':{}}
    usg_priors = {'usg':{},'2pt':{},'3pt':{},'fg':{}}
    for i in (players):
        for key in p_priors:
            p_priors[key][i] = [0,0]
        for key in usg_priors:
            usg_priors[key][i] = [0,0]
    for i in teams:
        for key in t_priors:
            t_priors[key][i] = [0,0]
    
    #use arma method for these
    for i in teams:
        t_priors['oreb_pct'][i] = 0.3
        t_priors['dreb_pct'][i] = 0.7
    for i in players:
        usg_priors['usg'][i] = 0.2


    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}



    #Using optimal values determined in other_latent_models.py
    fatten = {'3pt_pct_off':0.997,'2pt_pct_off':0.99,'ft_pct':0.98,'to_pct_off':0.98,"ftr_off":0.97,'3pt_pct_def':(0.995,1),'2pt_pct_def':(0.974,0.5),
              'to_pct_def':(0.96,0.8),"ftr_def":(0.96,0.7),'oreb_pct':0.05,'dreb_pct':0.05,'usg':0.2,'usg_3pt':(0.85,0.2),'usg_2pt':(0.85,0.2),'usg_fg':(0.85,0.2)}

    last_season = -1
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            team_adv_game = team_adv_bs.loc[team_adv_bs["game_id"] == gid,].reset_index()
            cur_usg = usg_bs.loc[usg_bs["game_id"] == gid,].dropna().reset_index()
            
            if (last_season != -1 and cur_season != last_season):
                for team in teams:
                    for j in range(2):
                        t_priors['3pt_pct_def'][team][j] *= fatten['3pt_pct_def'][1]
                        t_priors['2pt_pct_def'][team][j] *= fatten['2pt_pct_def'][1]
                        t_priors['to_pct_def'][team][j] *= fatten['to_pct_def'][1]
                        t_priors['ftr_def'][team][j] *= fatten['ftr_def'][1]
                for player in players:
                    for j in range(2):
                        usg_priors['3pt'][player][j] *= fatten['usg_3pt'][1]
                        usg_priors['2pt'][player][j] *= fatten['usg_2pt'][1]
                        usg_priors['fg'][player][j] *= fatten['usg_fg'][1]
            last_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
                h_eff = team_adv_game.at[0,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            
            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec

            


            for x in [h_id, a_id]:
                #for usg updates
                total_2pt_attempts = cur_game.loc[cur_game['team_id'] == x, ]['fieldGoalsAttempted'].sum() - cur_game.loc[cur_game['team_id'] == x, ]['threePointersAttempted'].sum()
                total_3pt_attempts = cur_game.loc[cur_game['team_id'] == x, ]['threePointersAttempted'].sum()
                total_fg_attempts = cur_game.loc[cur_game['team_id'] == x, ]['fieldGoalsAttempted'].sum()

                if (x == h_id):
                    y = a_id
                else:
                    y = h_id
                
                cur_f = {'game_id':team_game.at[0,'game_id'],'date':date,'season':cur_season,'off_team_id':x,'def_team_id':y}

                for yyy in ['last_','cur_']:
                    for key in p_priors:
                        if (key not in ['ftr_off','ft_pct']):
                            cur_f[yyy+'pred_'+key] = 0
                    cur_f[yyy+'pred_ftrXftp_off'] = 0
                    cur_f[yyy+'pred_ftrXftp_def'] = 0

                if (x == h_id):
                    cur_f['home_off'] = 1
                else:
                    cur_f['home_off'] = 0

                last_usage_time_total = 0
                last_3pt_usg_total = 0
                last_2pt_usg_total = 0
                last_fg_usg_total = 0
                for player in last_lineup[x]:
                    last_usage_time_total += last_lineup[x][player] * usg_priors['usg'][player]
                    if (usg_priors['3pt'][player][0] + usg_priors['3pt'][player][1] > 0):
                        last_3pt_usg_total += usg_priors['3pt'][player][0] / (usg_priors['3pt'][player][0] + usg_priors['3pt'][player][1])
                    if (usg_priors['2pt'][player][0] + usg_priors['2pt'][player][1] > 0):
                        last_2pt_usg_total += usg_priors['2pt'][player][0] / (usg_priors['2pt'][player][0] + usg_priors['2pt'][player][1])
                    if (usg_priors['fg'][player][0] + usg_priors['fg'][player][1] > 0):
                        last_fg_usg_total += usg_priors['fg'][player][0] / (usg_priors['fg'][player][0] + usg_priors['fg'][player][1])

                for player in last_lineup[x]:
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0] > 0):
                        usg_weight = (usg_priors['3pt'][player][0] / (usg_priors['3pt'][player][0] + usg_priors['3pt'][player][1])) / last_3pt_usg_total
                        cur_f['last_pred_3pt_pct_off'] += usg_weight * (p_priors['3pt_pct_off'][player][0] / (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0]))
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0] > 0):
                        usg_weight = (usg_priors['2pt'][player][0] / (usg_priors['2pt'][player][0] + usg_priors['2pt'][player][1])) / last_2pt_usg_total
                        cur_f['last_pred_2pt_pct_off'] += usg_weight * (p_priors['2pt_pct_off'][player][0] / (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0]))
                    #to_pct_off
                    if (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0] > 0):
                        usg_weight = last_lineup[x][player] * usg_priors['usg'][player] / last_usage_time_total
                        cur_f['last_pred_to_pct_off'] += usg_weight * (p_priors['to_pct_off'][player][0] / (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0]))
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0] > 0):
                        usg_weight = (usg_priors['fg'][player][0] / (usg_priors['fg'][player][0] + usg_priors['fg'][player][1])) / last_fg_usg_total
                        cur_f['last_pred_ftrXftp_off'] += usg_weight * (p_priors['ftr_off'][player][0] / (p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        usg_weight = (usg_priors['fg'][player][0] / (usg_priors['fg'][player][0] + usg_priors['fg'][player][1])) / last_fg_usg_total
                        cur_f['last_pred_ftrXftp_def'] += usg_weight * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))

                cur_usage_time_total = 0
                cur_3pt_usg_total = 0
                cur_2pt_usg_total = 0
                cur_fg_usg_total = 0
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_usage_time_total += cur_lineup[x][row['player_id']] * usg_priors['usg'][row['player_id']]
                    if (usg_priors['3pt'][row['player_id']][0] + usg_priors['3pt'][row['player_id']][1] > 0):
                        cur_3pt_usg_total += usg_priors['3pt'][row['player_id']][0] / (usg_priors['3pt'][row['player_id']][0] + usg_priors['3pt'][row['player_id']][1])
                    if (usg_priors['2pt'][row['player_id']][0] + usg_priors['2pt'][row['player_id']][1] > 0):
                        cur_2pt_usg_total += usg_priors['2pt'][row['player_id']][0] / (usg_priors['2pt'][row['player_id']][0] + usg_priors['2pt'][row['player_id']][1])
                    if (usg_priors['fg'][row['player_id']][0] + usg_priors['fg'][row['player_id']][1] > 0):
                        cur_fg_usg_total += usg_priors['fg'][row['player_id']][0] / (usg_priors['fg'][row['player_id']][0] + usg_priors['fg'][row['player_id']][1])

                #player features
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0] > 0):
                        usg_weight = (usg_priors['3pt'][row['player_id']][0] / (usg_priors['3pt'][row['player_id']][0] + usg_priors['3pt'][row['player_id']][1])) / cur_3pt_usg_total
                        cur_f['cur_pred_3pt_pct_off'] += usg_weight * (p_priors['3pt_pct_off'][row['player_id']][0] / (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0]))
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0] > 0):
                        usg_weight = (usg_priors['2pt'][row['player_id']][0] / (usg_priors['2pt'][row['player_id']][0] + usg_priors['2pt'][row['player_id']][1])) / cur_2pt_usg_total
                        cur_f['cur_pred_2pt_pct_off'] += usg_weight * (p_priors['2pt_pct_off'][row['player_id']][0] / (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0]))
                    #to_pct_off
                    if (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0] > 0):
                        usg_weight = cur_lineup[x][row['player_id']] * usg_priors['usg'][row['player_id']] / cur_usage_time_total
                        cur_f['cur_pred_to_pct_off'] += usg_weight * (p_priors['to_pct_off'][row['player_id']][0] / (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0]))
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0] > 0):
                        usg_weight = (usg_priors['fg'][row['player_id']][0] / (usg_priors['fg'][row['player_id']][0] + usg_priors['fg'][row['player_id']][1])) / cur_fg_usg_total
                        cur_f['cur_pred_ftrXftp_off'] += usg_weight * (p_priors['ftr_off'][row['player_id']][0] / (p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        usg_weight = (usg_priors['fg'][row['player_id']][0] / (usg_priors['fg'][row['player_id']][0] + usg_priors['fg'][row['player_id']][1])) / cur_fg_usg_total
                        cur_f['cur_pred_ftrXftp_def'] += usg_weight * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))


                    #Update player stuff
                    p_priors['3pt_pct_off'][row['player_id']][0] += row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][1] += row['threePointersAttempted'] - row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][0] *= fatten['3pt_pct_off']
                    p_priors['3pt_pct_off'][row['player_id']][1] *= fatten['3pt_pct_off']

                    p_priors['2pt_pct_off'][row['player_id']][0] += (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][1] += (row['fieldGoalsAttempted'] - row['threePointersAttempted']) - (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][0] *= fatten['2pt_pct_off']
                    p_priors['2pt_pct_off'][row['player_id']][1] *= fatten['2pt_pct_off']

                    p_priors['ft_pct'][row['player_id']][0] += row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][1] += row['freeThrowsAttempted'] - row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][0] *= fatten['ft_pct']
                    p_priors['ft_pct'][row['player_id']][1] *= fatten['ft_pct']

                    p_priors['to_pct_off'][row['player_id']][0] += row['turnovers']
                    p_priors['to_pct_off'][row['player_id']][1] += row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists']
                    p_priors['to_pct_off'][row['player_id']][0] *= fatten['to_pct_off']
                    p_priors['to_pct_off'][row['player_id']][1] *= fatten['to_pct_off']

                    p_priors['ftr_off'][row['player_id']][0] += row['freeThrowsAttempted']
                    p_priors['ftr_off'][row['player_id']][1] += row['fieldGoalsAttempted']
                    p_priors['ftr_off'][row['player_id']][0] *= fatten['ftr_off']
                    p_priors['ftr_off'][row['player_id']][1] *= fatten['ftr_off']

                    usg_priors['2pt'][row['player_id']][0] += row['fieldGoalsAttempted'] - row['threePointersAttempted']
                    usg_priors['2pt'][row['player_id']][1] += total_2pt_attempts - (row['fieldGoalsAttempted'] - row['threePointersAttempted'])
                    usg_priors['2pt'][row['player_id']][0] *= fatten['usg_2pt'][0]
                    usg_priors['2pt'][row['player_id']][1] *= fatten['usg_2pt'][0]

                    usg_priors['3pt'][row['player_id']][0] += row['threePointersAttempted']
                    usg_priors['3pt'][row['player_id']][1] += total_3pt_attempts - row['threePointersAttempted']
                    usg_priors['3pt'][row['player_id']][0] *= fatten['usg_3pt'][0]
                    usg_priors['3pt'][row['player_id']][1] *= fatten['usg_3pt'][0]

                    usg_priors['fg'][row['player_id']][0] += row['fieldGoalsAttempted']
                    usg_priors['fg'][row['player_id']][1] += total_fg_attempts - row['fieldGoalsAttempted']
                    usg_priors['fg'][row['player_id']][0] *= fatten['usg_fg'][0]
                    usg_priors['fg'][row['player_id']][1] *= fatten['usg_fg'][0]
                
                #update for overall usage
                for index, row in cur_usg.loc[cur_usg['team_id'] == x,].iterrows():
                    time_played = 5 * row['seconds'] / total_sec
                    usg_priors['usg'][row['player_id']] += (row['usagePercentage'] - usg_priors['usg'][row['player_id']]) * fatten['usg'] * time_played
                #team features
                if (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0] > 0):
                    cur_f['pred_3pt_pct_def'] = (t_priors['3pt_pct_def'][y][0] / (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0]))
                if (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0] > 0):
                    cur_f['pred_2pt_pct_def'] = (t_priors['2pt_pct_def'][y][0] / (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0]))
                if (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0] > 0):
                    cur_f['pred_to_pct_def'] = (t_priors['to_pct_def'][y][0] / (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0]))
                cur_f['pred_oreb_pct_off'] = t_priors['oreb_pct'][x]
                cur_f['pred_oreb_pct_def'] = 1 - t_priors['dreb_pct'][y]

                if (x == h_id):
                    ind = 0
                    rev_ind = 1
                else:
                    ind = 1
                    rev_ind = 0
                
                #Update
                t_priors['3pt_pct_def'][y][0] += team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][1] += team_game.at[ind,'threePointersAttempted'] - team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][0] *= fatten['3pt_pct_def'][0]
                t_priors['3pt_pct_def'][y][1] *= fatten['3pt_pct_def'][0]

                t_priors['2pt_pct_def'][y][0] += team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade']
                t_priors['2pt_pct_def'][y][1] += (team_game.at[ind,'fieldGoalsAttempted'] - team_game.at[ind,'threePointersAttempted']) - (team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade'])
                t_priors['2pt_pct_def'][y][0] *= fatten['2pt_pct_def'][0]
                t_priors['2pt_pct_def'][y][1] *= fatten['2pt_pct_def'][0]

                t_priors['to_pct_def'][y][0] += team_game.at[ind,'turnovers']
                t_priors['to_pct_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted'] + team_game.at[ind,'freeThrowsAttempted']*0.44
                t_priors['to_pct_def'][y][0] *= fatten['to_pct_def'][0]
                t_priors['to_pct_def'][y][1] *= fatten['to_pct_def'][0]

                t_priors['ftr_def'][y][0] += team_game.at[ind,'freeThrowsAttempted']
                t_priors['ftr_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted']
                t_priors['ftr_def'][y][0] *= fatten['ftr_def'][0]
                t_priors['ftr_def'][y][1] *= fatten['ftr_def'][0]

                pred_oreb_pct = (t_priors['oreb_pct'][x] + (1 - t_priors['dreb_pct'][y])) / 2
                actual_oreb_pct = team_game.at[ind,'reboundsOffensive'] / (team_game.at[ind,'reboundsOffensive'] + team_game.at[rev_ind,'reboundsDefensive'])
                t_priors['oreb_pct'][x] += (actual_oreb_pct - pred_oreb_pct) * fatten['oreb_pct']
                t_priors['dreb_pct'][y] += ((1-actual_oreb_pct) - (1-pred_oreb_pct)) * fatten['dreb_pct']
                
                cur_f['actual_eff'] = team_adv_game.at[ind,'offensiveRating']

                if (cur_season != "1996-97"):
                    features.append(cur_f)

            for x in [h_id, a_id]:
                last_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    last_lineup[x][row['player_id']] = row['seconds']/total_sec

    
    fdf = pd.DataFrame(features)

    fdf.to_csv("./intermediates/regression_formatted/second.csv", index=False)

#first but with some new features
def third():
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    team_adv_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    for i in range(len(player_bs.index)):
        if (not pd.isnull(player_bs.at[i,'minutes']) and ':' not in player_bs.at[i,'minutes']):
            player_bs.at[i,'minutes'] = player_bs.at[i,'minutes'] + ':00:00'

            
    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    team_schedule = {}
    for team in teams:
        team_schedule[team] = list(games.loc[games['h_team_id']==team,]['game_date']) + list(games.loc[games['a_team_id']==team,]['game_date'])
        team_schedule[team].sort()

    features = []

    p_priors = {'3pt_pct_off':{},'2pt_pct_off':{},'ft_pct':{},'to_pct_off':{},"ftr_off":{}}
    t_priors = {'3pt_pct_def':{},'2pt_pct_def':{},'to_pct_def':{},"ftr_def":{},'oreb_pct':{},'dreb_pct':{}}
    p_form = {'3pt_pct_off':{},'2pt_pct_off':{},'ft_pct':{},'to_pct_off':{},"ftr_off":{}}
    t_form = {'3pt_pct_def':{},'2pt_pct_def':{},'to_pct_def':{},"ftr_def":{},'oreb_pct':{},'dreb_pct':{}}
    for i in (players):
        for key in p_priors:
            p_priors[key][i] = [0,0]
            p_form[key][i] = [[],[]]
    for i in teams:
        for key in t_priors:
            t_priors[key][i] = [0,0]
            t_form[key][i] = []
    
    #use arma method for these
    for i in teams:
        t_priors['oreb_pct'][i] = 0.3
        t_priors['dreb_pct'][i] = 0.7

    #matchup form
    mu_form = {'3pt_pct':{},'2pt_pct':{},'to_pct':{},'ftr':{},'oreb_pct':{},'eff':{}}
    for i in teams:
        for j in teams:
            if (i < j):
                for key in mu_form:
                    #first list is for team i and second is for team j
                    mu_form[key][str(i)+'/'+str(j)] = [[],[]]


    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}



    #Using optimal values determined in other_latent_models.py
    fatten = {'3pt_pct_off':0.997,'2pt_pct_off':0.99,'ft_pct':0.98,'to_pct_off':0.98,"ftr_off":0.97,'3pt_pct_def':(0.995,1),'2pt_pct_def':(0.974,0.5),
              'to_pct_def':(0.96,0.8),"ftr_def":(0.96,0.7),'oreb_pct':0.05,'dreb_pct':0.05}

    last_season = -1
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            team_adv_game = team_adv_bs.loc[team_adv_bs["game_id"] == gid,].reset_index()
            
            if (last_season != -1 and cur_season != last_season):
                for team in teams:
                    for j in range(2):
                        t_priors['3pt_pct_def'][team][j] *= fatten['3pt_pct_def'][1]
                        t_priors['2pt_pct_def'][team][j] *= fatten['2pt_pct_def'][1]
                        t_priors['to_pct_def'][team][j] *= fatten['to_pct_def'][1]
                        t_priors['ftr_def'][team][j] *= fatten['ftr_def'][1]
            last_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
                h_eff = team_adv_game.at[0,'offensiveRating']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            for x in [h_id, a_id]:
                cur_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    cur_lineup[x][row['player_id']] = row['seconds']/total_sec


            for x in [h_id, a_id]:
                if (x == h_id):
                    y = a_id
                else:
                    y = h_id
                
                cur_f = {'game_id':team_game.at[0,'game_id'],'date':date,'season':cur_season,'off_team_id':x,'def_team_id':y,
                         'off_b2b_leg_1':0,'off_b2b_leg_2':0,'def_b2b_leg_1':0,'def_b2b_leg_2':0}

                if (date + datetime.timedelta(days=1) in team_schedule[x]):
                    cur_f['off_b2b_leg_1'] = 1
                if (date - datetime.timedelta(days=1) in team_schedule[x]):
                    cur_f['off_b2b_leg_2'] = 1

                if (date + datetime.timedelta(days=1) in team_schedule[y]):
                    cur_f['def_b2b_leg_1'] = 1
                if (date - datetime.timedelta(days=1) in team_schedule[y]):
                    cur_f['def_b2b_leg_2'] = 1

                #for scaling up the stats in the case when some players cant have their contribution count and everyone else's contribution needs to count for more than their playing time would allow
                tracker = {}

                for yyy in ['last_','cur_']:
                    for key in p_priors:
                        if (key not in ['ftr_off','ft_pct']):
                            cur_f[yyy+'pred_'+key] = 0
                            cur_f[yyy+'pred_'+key+'_last_5'] = 0
                            cur_f[yyy+'pred_'+key+'_last_10'] = 0
                            tracker[yyy+'pred_'+key] = 0
                            tracker[yyy+'pred_'+key+'_last_5'] = 0
                            tracker[yyy+'pred_'+key+'_last_10'] = 0
                    cur_f[yyy+'pred_ftrXftp_off'] = 0
                    cur_f[yyy+'pred_ftrXftp_def'] = 0
                    cur_f[yyy+'pred_ftrXftp_off'+'_last_5'] = 0
                    cur_f[yyy+'pred_ftrXftp_def'+'_last_5'] = 0
                    cur_f[yyy+'pred_ftrXftp_off'+'_last_10'] = 0
                    cur_f[yyy+'pred_ftrXftp_def'+'_last_10'] = 0
                    tracker[yyy+'pred_ftrXftp_off'] = 0
                    tracker[yyy+'pred_ftrXftp_def'] = 0
                    tracker[yyy+'pred_ftrXftp_off'+'_last_5'] = 0
                    tracker[yyy+'pred_ftrXftp_def'+'_last_5'] = 0
                    tracker[yyy+'pred_ftrXftp_off'+'_last_10'] = 0
                    tracker[yyy+'pred_ftrXftp_def'+'_last_10'] = 0

                if (x == h_id):
                    cur_f['home_off'] = 1
                else:
                    cur_f['home_off'] = 0

                for player in last_lineup[x]:
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0] > 0):
                        cur_f['last_pred_3pt_pct_off'] += last_lineup[x][player] * (p_priors['3pt_pct_off'][player][0] / (p_priors['3pt_pct_off'][player][1] + p_priors['3pt_pct_off'][player][0]))
                        tracker['last_pred_3pt_pct_off'] += last_lineup[x][player]
                        if (len(p_form['3pt_pct_off'][player][0]) >= 10):
                            cur_f['last_pred_3pt_pct_off_last_10'] += last_lineup[x][player] * (np.sum(p_form['3pt_pct_off'][player][0][-10:]) /  np.sum(p_form['3pt_pct_off'][player][1][-10:]))
                            tracker['last_pred_3pt_pct_off_last_10'] += last_lineup[x][player]
                        if (len(p_form['3pt_pct_off'][player][0]) >= 5):
                            cur_f['last_pred_3pt_pct_off_last_5'] += last_lineup[x][player] * (np.sum(p_form['3pt_pct_off'][player][0][-5:]) /  np.sum(p_form['3pt_pct_off'][player][1][-5:]))
                            tracker['last_pred_3pt_pct_off_last_5'] += last_lineup[x][player]
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0] > 0):
                        cur_f['last_pred_2pt_pct_off'] += last_lineup[x][player] * (p_priors['2pt_pct_off'][player][0] / (p_priors['2pt_pct_off'][player][1] + p_priors['2pt_pct_off'][player][0]))
                        tracker['last_pred_2pt_pct_off'] += last_lineup[x][player]
                        if (len(p_form['2pt_pct_off'][player][0]) >= 10):
                            cur_f['last_pred_2pt_pct_off_last_10'] += last_lineup[x][player] * (np.sum(p_form['2pt_pct_off'][player][0][-10:]) /  np.sum(p_form['2pt_pct_off'][player][1][-10:]))
                            tracker['last_pred_2pt_pct_off_last_10'] += last_lineup[x][player]
                        if (len(p_form['2pt_pct_off'][player][0]) >= 5):
                            cur_f['last_pred_2pt_pct_off_last_5'] += last_lineup[x][player] * (np.sum(p_form['2pt_pct_off'][player][0][-5:]) /  np.sum(p_form['2pt_pct_off'][player][1][-5:]))
                            tracker['last_pred_2pt_pct_off_last_5'] += last_lineup[x][player]
                    #to_pct_off
                    if (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0] > 0):
                        cur_f['last_pred_to_pct_off'] += last_lineup[x][player] * (p_priors['to_pct_off'][player][0] / (p_priors['to_pct_off'][player][1] + p_priors['to_pct_off'][player][0]))
                        tracker['last_pred_to_pct_off'] += last_lineup[x][player]
                        if (len(p_form['to_pct_off'][player][0]) >= 10):
                            cur_f['last_pred_to_pct_off_last_10'] += last_lineup[x][player] * (np.sum(p_form['to_pct_off'][player][0][-10:]) /  np.sum(p_form['to_pct_off'][player][1][-10:]))
                            tracker['last_pred_to_pct_off_last_10'] += last_lineup[x][player]
                        if (len(p_form['to_pct_off'][player][0]) >= 5):
                            cur_f['last_pred_to_pct_off_last_5'] += last_lineup[x][player] * (np.sum(p_form['to_pct_off'][player][0][-5:]) /  np.sum(p_form['to_pct_off'][player][1][-5:]))
                            tracker['last_pred_to_pct_off_last_5'] += last_lineup[x][player]
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0] > 0):
                        cur_f['last_pred_ftrXftp_off'] += last_lineup[x][player] * (p_priors['ftr_off'][player][0] / (p_priors['ftr_off'][player][1] + p_priors['ftr_off'][player][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))
                        tracker['last_pred_ftrXftp_off'] += last_lineup[x][player]
                        if (len(p_form['ftr_off'][player][0]) >= 10):
                            cur_f['last_pred_ftrXftp_off_last_10'] += last_lineup[x][player] * (np.sum(p_form['ftr_off'][player][0][-10:]) /  np.sum(p_form['ftr_off'][player][1][-10:])) * (np.sum(p_form['ft_pct'][player][0][-10:]) / np.sum(p_form['ft_pct'][player][1][-10:]))
                            tracker['last_pred_ftrXftp_off_last_10'] += last_lineup[x][player]
                        if (len(p_form['ftr_off'][player][0]) >= 5):
                            cur_f['last_pred_ftrXftp_off_last_5'] += last_lineup[x][player] * (np.sum(p_form['ftr_off'][player][0][-5:]) /  np.sum(p_form['ftr_off'][player][1][-5:])) * (np.sum(p_form['ft_pct'][player][0][-5:]) / np.sum(p_form['ft_pct'][player][1][-5:]))
                            tracker['last_pred_ftrXftp_off_last_5'] += last_lineup[x][player]
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        cur_f['last_pred_ftrXftp_def'] += last_lineup[x][player] * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][player][0] / (p_priors['ft_pct'][player][1] + p_priors['ft_pct'][player][0]))
                        tracker['last_pred_ftrXftp_def'] += last_lineup[x][player]
                        if (len(p_form['ft_pct'][player][0]) >= 10 and len(t_form['ftr_def'][y]) >= 10):
                            cur_f['last_pred_ftrXftp_def_last_10'] += last_lineup[x][player] * np.average(t_form['ftr_def'][y][-10:]) * (np.sum(p_form['ft_pct'][player][0][-10:]) / np.sum(p_form['ft_pct'][player][1][-10:]))
                            tracker['last_pred_ftrXftp_def_last_10'] += last_lineup[x][player]
                        if (len(p_form['ft_pct'][player][0]) >= 5 and len(t_form['ftr_def'][y]) >= 5):
                            cur_f['last_pred_ftrXftp_def_last_5'] += last_lineup[x][player] * np.average(t_form['ftr_def'][y][-5:]) * (np.sum(p_form['ft_pct'][player][0][-5:]) / np.sum(p_form['ft_pct'][player][1][-5:]))
                            tracker['last_pred_ftrXftp_def_last_5'] += last_lineup[x][player]

                #player features
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    #3pt_pct_off
                    if (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_3pt_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['3pt_pct_off'][row['player_id']][0] / (p_priors['3pt_pct_off'][row['player_id']][1] + p_priors['3pt_pct_off'][row['player_id']][0]))
                        tracker['cur_pred_3pt_pct_off'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['3pt_pct_off'][row['player_id']][0]) >= 10):
                            cur_f['cur_pred_3pt_pct_off_last_10'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['3pt_pct_off'][row['player_id']][0][-10:]) /  np.sum(p_form['3pt_pct_off'][row['player_id']][1][-10:]))
                            tracker['cur_pred_3pt_pct_off_last_10'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['3pt_pct_off'][row['player_id']][0]) >= 5):
                            cur_f['cur_pred_3pt_pct_off_last_5'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['3pt_pct_off'][row['player_id']][0][-5:]) /  np.sum(p_form['3pt_pct_off'][row['player_id']][1][-5:]))
                            tracker['cur_pred_3pt_pct_off_last_5'] += cur_lineup[x][row['player_id']]
                    #2pt_pct_off
                    if (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_2pt_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['2pt_pct_off'][row['player_id']][0] / (p_priors['2pt_pct_off'][row['player_id']][1] + p_priors['2pt_pct_off'][row['player_id']][0]))
                        tracker['cur_pred_2pt_pct_off'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['2pt_pct_off'][row['player_id']][0]) >= 10):
                            cur_f['cur_pred_2pt_pct_off_last_10'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['2pt_pct_off'][row['player_id']][0][-10:]) /  np.sum(p_form['2pt_pct_off'][row['player_id']][1][-10:]))
                            tracker['cur_pred_2pt_pct_off_last_10'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['2pt_pct_off'][row['player_id']][0]) >= 5):
                            cur_f['cur_pred_2pt_pct_off_last_5'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['2pt_pct_off'][row['player_id']][0][-5:]) /  np.sum(p_form['2pt_pct_off'][row['player_id']][1][-5:]))
                            tracker['cur_pred_2pt_pct_off_last_5'] += cur_lineup[x][row['player_id']]
                    #to_pct_off
                    if (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_to_pct_off'] += cur_lineup[x][row['player_id']] * (p_priors['to_pct_off'][row['player_id']][0] / (p_priors['to_pct_off'][row['player_id']][1] + p_priors['to_pct_off'][row['player_id']][0]))
                        tracker['cur_pred_to_pct_off'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['to_pct_off'][row['player_id']][0]) >= 10):
                            cur_f['cur_pred_to_pct_off_last_10'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['to_pct_off'][row['player_id']][0][-10:]) /  np.sum(p_form['to_pct_off'][row['player_id']][1][-10:]))
                            tracker['cur_pred_to_pct_off_last_10'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['to_pct_off'][row['player_id']][0]) >= 5):
                            cur_f['cur_pred_to_pct_off_last_5'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['to_pct_off'][row['player_id']][0][-5:]) /  np.sum(p_form['to_pct_off'][row['player_id']][1][-5:]))
                            tracker['cur_pred_to_pct_off_last_5'] += cur_lineup[x][row['player_id']]
                    #free throw pct _ ftr offense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0] > 0):
                        cur_f['cur_pred_ftrXftp_off'] += cur_lineup[x][row['player_id']] * (p_priors['ftr_off'][row['player_id']][0] / (p_priors['ftr_off'][row['player_id']][1] + p_priors['ftr_off'][row['player_id']][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))
                        tracker['cur_pred_ftrXftp_off'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['ftr_off'][row['player_id']][0]) >= 10):
                            cur_f['cur_pred_ftrXftp_off_last_10'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['ftr_off'][row['player_id']][0][-10:]) /  np.sum(p_form['ftr_off'][row['player_id']][1][-10:])) * (np.sum(p_form['ft_pct'][row['player_id']][0][-10:]) / np.sum(p_form['ft_pct'][row['player_id']][1][-10:]))
                            tracker['cur_pred_ftrXftp_off_last_10'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['ftr_off'][row['player_id']][0]) >= 5):
                            cur_f['cur_pred_ftrXftp_off_last_5'] += cur_lineup[x][row['player_id']] * (np.sum(p_form['ftr_off'][row['player_id']][0][-5:]) /  np.sum(p_form['ftr_off'][row['player_id']][1][-5:])) * (np.sum(p_form['ft_pct'][row['player_id']][0][-5:]) / np.sum(p_form['ft_pct'][row['player_id']][1][-5:]))
                            tracker['cur_pred_ftrXftp_off_last_5'] += cur_lineup[x][row['player_id']]
                    #free throw pct _ ftr defense interaction
                    if (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0] > 0 and t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0] > 0):
                        cur_f['cur_pred_ftrXftp_def'] += cur_lineup[x][row['player_id']] * (t_priors['ftr_def'][y][0] / (t_priors['ftr_def'][y][1] + t_priors['ftr_def'][y][0])) * (p_priors['ft_pct'][row['player_id']][0] / (p_priors['ft_pct'][row['player_id']][1] + p_priors['ft_pct'][row['player_id']][0]))
                        tracker['cur_pred_ftrXftp_def'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['ft_pct'][row['player_id']][0]) >= 10 and len(t_form['ftr_def'][y]) >= 10):
                            cur_f['cur_pred_ftrXftp_def_last_10'] += cur_lineup[x][row['player_id']] * np.average(t_form['ftr_def'][y][-10:]) * (np.sum(p_form['ft_pct'][row['player_id']][0][-10:]) / np.sum(p_form['ft_pct'][row['player_id']][1][-10:]))
                            tracker['cur_pred_ftrXftp_def_last_10'] += cur_lineup[x][row['player_id']]
                        if (len(p_form['ft_pct'][row['player_id']][0]) >= 5 and len(t_form['ftr_def'][y]) >= 5):
                            cur_f['cur_pred_ftrXftp_def_last_5'] += cur_lineup[x][row['player_id']] * np.average(t_form['ftr_def'][y][-5:]) * (np.sum(p_form['ft_pct'][row['player_id']][0][-5:]) / np.sum(p_form['ft_pct'][row['player_id']][1][-5:]))
                            tracker['cur_pred_ftrXftp_def_last_5'] += cur_lineup[x][row['player_id']]
                    
                    #Update player stuff
                    p_priors['3pt_pct_off'][row['player_id']][0] += row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][1] += row['threePointersAttempted'] - row['threePointersMade']
                    p_priors['3pt_pct_off'][row['player_id']][0] *= fatten['3pt_pct_off']
                    p_priors['3pt_pct_off'][row['player_id']][1] *= fatten['3pt_pct_off']

                    p_priors['2pt_pct_off'][row['player_id']][0] += (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][1] += (row['fieldGoalsAttempted'] - row['threePointersAttempted']) - (row['fieldGoalsMade'] - row['threePointersMade'])
                    p_priors['2pt_pct_off'][row['player_id']][0] *= fatten['2pt_pct_off']
                    p_priors['2pt_pct_off'][row['player_id']][1] *= fatten['2pt_pct_off']

                    p_priors['ft_pct'][row['player_id']][0] += row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][1] += row['freeThrowsAttempted'] - row['freeThrowsMade']
                    p_priors['ft_pct'][row['player_id']][0] *= fatten['ft_pct']
                    p_priors['ft_pct'][row['player_id']][1] *= fatten['ft_pct']

                    p_priors['to_pct_off'][row['player_id']][0] += row['turnovers']
                    p_priors['to_pct_off'][row['player_id']][1] += row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists']
                    p_priors['to_pct_off'][row['player_id']][0] *= fatten['to_pct_off']
                    p_priors['to_pct_off'][row['player_id']][1] *= fatten['to_pct_off']

                    p_priors['ftr_off'][row['player_id']][0] += row['freeThrowsAttempted']
                    p_priors['ftr_off'][row['player_id']][1] += row['fieldGoalsAttempted']
                    p_priors['ftr_off'][row['player_id']][0] *= fatten['ftr_off']
                    p_priors['ftr_off'][row['player_id']][1] *= fatten['ftr_off']

                    if (row['threePointersAttempted'] > 0):
                        p_form['3pt_pct_off'][row['player_id']][0].append(row['threePointersMade'])
                        p_form['3pt_pct_off'][row['player_id']][1].append(row['threePointersAttempted'])
                    
                    if ((row['fieldGoalsAttempted'] - row['threePointersAttempted']) > 0):
                        p_form['2pt_pct_off'][row['player_id']][0].append((row['fieldGoalsMade'] - row['threePointersMade']))
                        p_form['2pt_pct_off'][row['player_id']][1].append((row['fieldGoalsAttempted'] - row['threePointersAttempted']))

                    if (row['freeThrowsAttempted'] > 0):
                        p_form['ft_pct'][row['player_id']][0].append(row['freeThrowsMade'])
                        p_form['ft_pct'][row['player_id']][1].append(row['freeThrowsAttempted'])

                    if (row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists'] + row['turnovers'] > 0):
                        p_form['to_pct_off'][row['player_id']][0].append(row['turnovers'])
                        p_form['to_pct_off'][row['player_id']][1].append(row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists'] + row['turnovers'])

                    if (row['fieldGoalsAttempted'] + row['freeThrowsAttempted'] > 0):
                        p_form['ftr_off'][row['player_id']][0].append(row['freeThrowsAttempted'])
                        p_form['ftr_off'][row['player_id']][1].append(row['fieldGoalsAttempted'] + row['freeThrowsAttempted'])

                #team features
                if (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0] > 0):
                    cur_f['pred_3pt_pct_def'] = (t_priors['3pt_pct_def'][y][0] / (t_priors['3pt_pct_def'][y][1] + t_priors['3pt_pct_def'][y][0]))
                    if (len(t_form['3pt_pct_def'][y]) >= 10):
                        cur_f['pred_3pt_pct_def_last_10'] = np.average(t_form['3pt_pct_def'][y][-10:])
                    if (len(t_form['3pt_pct_def'][y]) >= 5):
                        cur_f['pred_3pt_pct_def_last_5'] = np.average(t_form['3pt_pct_def'][y][-5:])
                if (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0] > 0):
                    cur_f['pred_2pt_pct_def'] = (t_priors['2pt_pct_def'][y][0] / (t_priors['2pt_pct_def'][y][1] + t_priors['2pt_pct_def'][y][0]))
                    if (len(t_form['2pt_pct_def'][y]) >= 10):
                        cur_f['pred_2pt_pct_def_last_10'] = np.average(t_form['2pt_pct_def'][y][-10:])
                    if (len(t_form['2pt_pct_def'][y]) >= 5):
                        cur_f['pred_2pt_pct_def_last_5'] = np.average(t_form['2pt_pct_def'][y][-5:])
                if (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0] > 0):
                    cur_f['pred_to_pct_def'] = (t_priors['to_pct_def'][y][0] / (t_priors['to_pct_def'][y][1] + t_priors['to_pct_def'][y][0]))
                    if (len(t_form['to_pct_def'][y]) >= 10):
                        cur_f['pred_to_pct_def_last_10'] = np.average(t_form['to_pct_def'][y][-10:])
                    if (len(t_form['to_pct_def'][y]) >= 5):
                        cur_f['pred_to_pct_def_last_5'] = np.average(t_form['to_pct_def'][y][-5:])
                cur_f['pred_oreb_pct_off'] = t_priors['oreb_pct'][x]
                if (len(t_form['oreb_pct'][x]) >= 10):
                    cur_f['pred_oreb_pct_off_last_10'] = np.average(t_form['oreb_pct'][x][-10:])
                if (len(t_form['oreb_pct'][x]) >= 5):
                    cur_f['pred_oreb_pct_off_last_5'] = np.average(t_form['oreb_pct'][x][-5:])
                cur_f['pred_oreb_pct_def'] = 1 - t_priors['dreb_pct'][y]
                if (len(t_form['dreb_pct'][y]) >= 10):
                    cur_f['pred_oreb_pct_def_last_10'] = 1 - np.average(t_form['dreb_pct'][y][-10:])
                if (len(t_form['dreb_pct'][y]) >= 5):
                    cur_f['pred_oreb_pct_def_last_5'] = 1 - np.average(t_form['dreb_pct'][y][-5:])

                if (x < y):
                    if (len(mu_form['3pt_pct'][str(x)+'/'+str(y)][0]) >= 3):
                        cur_f['pred_3pt_pct_matchup_form'] = np.average(mu_form['3pt_pct'][str(x)+'/'+str(y)][0][-3:])
                        cur_f['pred_2pt_pct_matchup_form'] = np.average(mu_form['2pt_pct'][str(x)+'/'+str(y)][0][-3:])
                        cur_f['pred_to_pct_matchup_form'] = np.average(mu_form['to_pct'][str(x)+'/'+str(y)][0][-3:])
                        cur_f['pred_ftr_matchup_form'] = np.average(mu_form['ftr'][str(x)+'/'+str(y)][0][-3:])
                        cur_f['pred_oreb_pct_matchup_form'] = np.average(mu_form['oreb_pct'][str(x)+'/'+str(y)][0][-3:])
                        cur_f['pred_rtg_matchup_form'] = np.average(mu_form['eff'][str(x)+'/'+str(y)][0][-3:])
                else:
                    if (len(mu_form['3pt_pct'][str(y)+'/'+str(x)][1]) >= 3):
                        cur_f['pred_3pt_pct_matchup_form'] = np.average(mu_form['3pt_pct'][str(y)+'/'+str(x)][1][-3:])
                        cur_f['pred_2pt_pct_matchup_form'] = np.average(mu_form['2pt_pct'][str(y)+'/'+str(x)][1][-3:])
                        cur_f['pred_to_pct_matchup_form'] = np.average(mu_form['to_pct'][str(y)+'/'+str(x)][1][-3:])
                        cur_f['pred_ftr_matchup_form'] = np.average(mu_form['ftr'][str(y)+'/'+str(x)][1][-3:])
                        cur_f['pred_oreb_pct_matchup_form'] = np.average(mu_form['oreb_pct'][str(y)+'/'+str(x)][1][-3:])
                        cur_f['pred_rtg_matchup_form'] = np.average(mu_form['eff'][str(y)+'/'+str(x)][1][-3:])

                if (x == h_id):
                    ind = 0
                    rev_ind = 1
                else:
                    ind = 1
                    rev_ind = 0
                
                #Update
                t_priors['3pt_pct_def'][y][0] += team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][1] += team_game.at[ind,'threePointersAttempted'] - team_game.at[ind,'threePointersMade']
                t_priors['3pt_pct_def'][y][0] *= fatten['3pt_pct_def'][0]
                t_priors['3pt_pct_def'][y][1] *= fatten['3pt_pct_def'][0]

                t_priors['2pt_pct_def'][y][0] += team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade']
                t_priors['2pt_pct_def'][y][1] += (team_game.at[ind,'fieldGoalsAttempted'] - team_game.at[ind,'threePointersAttempted']) - (team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade'])
                t_priors['2pt_pct_def'][y][0] *= fatten['2pt_pct_def'][0]
                t_priors['2pt_pct_def'][y][1] *= fatten['2pt_pct_def'][0]

                t_priors['to_pct_def'][y][0] += team_game.at[ind,'turnovers']
                t_priors['to_pct_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted'] + team_game.at[ind,'freeThrowsAttempted']*0.44
                t_priors['to_pct_def'][y][0] *= fatten['to_pct_def'][0]
                t_priors['to_pct_def'][y][1] *= fatten['to_pct_def'][0]

                t_priors['ftr_def'][y][0] += team_game.at[ind,'freeThrowsAttempted']
                t_priors['ftr_def'][y][1] += team_game.at[ind,'fieldGoalsAttempted']
                t_priors['ftr_def'][y][0] *= fatten['ftr_def'][0]
                t_priors['ftr_def'][y][1] *= fatten['ftr_def'][0]

                pred_oreb_pct = (t_priors['oreb_pct'][x] + (1 - t_priors['dreb_pct'][y])) / 2
                actual_oreb_pct = team_game.at[ind,'reboundsOffensive'] / (team_game.at[ind,'reboundsOffensive'] + team_game.at[rev_ind,'reboundsDefensive'])
                t_priors['oreb_pct'][x] += (actual_oreb_pct - pred_oreb_pct) * fatten['oreb_pct']
                t_priors['dreb_pct'][y] += ((1-actual_oreb_pct) - (1-pred_oreb_pct)) * fatten['dreb_pct']

                if (team_game.at[ind,'threePointersAttempted'] > 0):
                    t_form['3pt_pct_def'][y].append(team_game.at[ind,'threePointersMade'] / team_game.at[ind,'threePointersAttempted'])

                t_form['2pt_pct_def'][y].append((team_game.at[ind,'fieldGoalsMade'] - team_game.at[ind,'threePointersMade']) / (team_game.at[ind,'fieldGoalsAttempted'] - team_game.at[ind,'threePointersAttempted']))

                t_form['to_pct_def'][y].append(team_game.at[ind,'turnovers'] / (team_game.at[ind,'fieldGoalsAttempted'] + team_game.at[ind,'freeThrowsAttempted']*0.44 + team_game.at[ind,'turnovers']))

                t_form['ftr_def'][y].append(team_game.at[ind,'freeThrowsAttempted'] / (team_game.at[ind,'fieldGoalsAttempted'] + team_game.at[ind,'freeThrowsAttempted']))

                t_form['oreb_pct'][x].append(actual_oreb_pct)

                t_form['dreb_pct'][y].append(1-actual_oreb_pct)

                #a_id so it only updates once and it does it on the second loop
                if (x == a_id and x < y):
                    if (team_game.at[0,'threePointersAttempted'] > 0):
                        mu_form['3pt_pct'][str(x)+'/'+str(y)][1].append(team_game.at[0,'threePointersMade'] / team_game.at[0,'threePointersAttempted'])
                    if (team_game.at[1,'threePointersAttempted'] > 0):
                        mu_form['3pt_pct'][str(x)+'/'+str(y)][0].append(team_game.at[1,'threePointersMade'] / team_game.at[1,'threePointersAttempted'])

                    mu_form['2pt_pct'][str(x)+'/'+str(y)][1].append((team_game.at[0,'fieldGoalsMade'] - team_game.at[0,'threePointersMade']) / (team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted']))
                    mu_form['2pt_pct'][str(x)+'/'+str(y)][0].append((team_game.at[1,'fieldGoalsMade'] - team_game.at[1,'threePointersMade']) / (team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted']))

                    mu_form['to_pct'][str(x)+'/'+str(y)][1].append(team_game.at[0,'turnovers'] / (team_game.at[0,'fieldGoalsAttempted'] + 0.44*team_game.at[0,'freeThrowsAttempted'] + team_game.at[0,'turnovers']))
                    mu_form['to_pct'][str(x)+'/'+str(y)][0].append(team_game.at[1,'turnovers'] / (team_game.at[1,'fieldGoalsAttempted'] + 0.44*team_game.at[1,'freeThrowsAttempted'] + team_game.at[1,'turnovers']))

                    mu_form['ftr'][str(x)+'/'+str(y)][1].append(team_game.at[0,'freeThrowsAttempted'] / (team_game.at[0,'freeThrowsAttempted'] + team_game.at[0,'fieldGoalsAttempted']))
                    mu_form['ftr'][str(x)+'/'+str(y)][0].append(team_game.at[1,'freeThrowsAttempted'] / (team_game.at[1,'freeThrowsAttempted'] + team_game.at[1,'fieldGoalsAttempted']))

                    mu_form['oreb_pct'][str(x)+'/'+str(y)][1].append(team_game.at[0,'reboundsOffensive'] / (team_game.at[0,'reboundsOffensive'] + team_game.at[1,'reboundsDefensive']))
                    mu_form['oreb_pct'][str(x)+'/'+str(y)][0].append(team_game.at[1,'reboundsOffensive'] / (team_game.at[1,'reboundsOffensive'] + team_game.at[0,'reboundsDefensive']))

                    mu_form['eff'][str(x)+'/'+str(y)][1].append(team_adv_game.at[0,'offensiveRating'])
                    mu_form['eff'][str(x)+'/'+str(y)][0].append(team_adv_game.at[1,'offensiveRating'])
                elif (x == a_id and x > y):
                    if (team_game.at[0,'threePointersAttempted'] > 0):
                        mu_form['3pt_pct'][str(y)+'/'+str(x)][0].append(team_game.at[0,'threePointersMade'] / team_game.at[0,'threePointersAttempted'])
                    if (team_game.at[1,'threePointersAttempted'] > 0):
                        mu_form['3pt_pct'][str(y)+'/'+str(x)][1].append(team_game.at[1,'threePointersMade'] / team_game.at[1,'threePointersAttempted'])

                    mu_form['2pt_pct'][str(y)+'/'+str(x)][0].append((team_game.at[0,'fieldGoalsMade'] - team_game.at[0,'threePointersMade']) / (team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted']))
                    mu_form['2pt_pct'][str(y)+'/'+str(x)][1].append((team_game.at[1,'fieldGoalsMade'] - team_game.at[1,'threePointersMade']) / (team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted']))

                    mu_form['to_pct'][str(y)+'/'+str(x)][0].append(team_game.at[0,'turnovers'] / (team_game.at[0,'fieldGoalsAttempted'] + 0.44*team_game.at[0,'freeThrowsAttempted'] + team_game.at[0,'turnovers']))
                    mu_form['to_pct'][str(y)+'/'+str(x)][1].append(team_game.at[1,'turnovers'] / (team_game.at[1,'fieldGoalsAttempted'] + 0.44*team_game.at[1,'freeThrowsAttempted'] + team_game.at[1,'turnovers']))

                    mu_form['ftr'][str(y)+'/'+str(x)][0].append(team_game.at[0,'freeThrowsAttempted'] / (team_game.at[0,'freeThrowsAttempted'] + team_game.at[0,'fieldGoalsAttempted']))
                    mu_form['ftr'][str(y)+'/'+str(x)][1].append(team_game.at[1,'freeThrowsAttempted'] / (team_game.at[1,'freeThrowsAttempted'] + team_game.at[1,'fieldGoalsAttempted']))

                    mu_form['oreb_pct'][str(y)+'/'+str(x)][0].append(team_game.at[0,'reboundsOffensive'] / (team_game.at[0,'reboundsOffensive'] + team_game.at[1,'reboundsDefensive']))
                    mu_form['oreb_pct'][str(y)+'/'+str(x)][1].append(team_game.at[1,'reboundsOffensive'] / (team_game.at[1,'reboundsOffensive'] + team_game.at[0,'reboundsDefensive']))

                    mu_form['eff'][str(y)+'/'+str(x)][0].append(team_adv_game.at[0,'offensiveRating'])
                    mu_form['eff'][str(y)+'/'+str(x)][1].append(team_adv_game.at[1,'offensiveRating'])
                
                cur_f['actual_eff'] = team_adv_game.at[ind,'offensiveRating']

                if (cur_season != "1996-97"):
                    for yyy in ['last_','cur_']:
                        for key in p_priors:
                            if (key not in ['ftr_off','ft_pct']):
                                for xxx in ['','_last_5','_last_10']:
                                    if (tracker[yyy+'pred_'+key+xxx] == 0):
                                        cur_f[yyy+'pred_'+key+xxx] = np.nan
                                    else:
                                        cur_f[yyy+'pred_'+key+xxx] = cur_f[yyy+'pred_'+key+xxx] / tracker[yyy+'pred_'+key+xxx]
                        for xxx in ['','_last_5','_last_10']:
                            if (tracker[yyy+'pred_ftrXftp_off'+xxx] == 0):
                                cur_f[yyy+'pred_ftrXftp_off'+xxx] = np.nan
                            else:
                                cur_f[yyy+'pred_ftrXftp_off'+xxx] = cur_f[yyy+'pred_ftrXftp_off'+xxx] / tracker[yyy+'pred_ftrXftp_off'+xxx]
                            if (tracker[yyy+'pred_ftrXftp_def'+xxx] == 0):
                                cur_f[yyy+'pred_ftrXftp_def'+xxx] = np.nan
                            else:
                                cur_f[yyy+'pred_ftrXftp_def'+xxx] = cur_f[yyy+'pred_ftrXftp_def'+xxx] / tracker[yyy+'pred_ftrXftp_def'+xxx]
                
                    features.append(cur_f)

            for x in [h_id, a_id]:
                last_lineup[x] = {}
                for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
                    last_lineup[x][row['player_id']] = row['seconds']/total_sec

    
    fdf = pd.DataFrame(features)

    fdf.to_csv("./intermediates/regression_formatted/third.csv", index=False)

third()