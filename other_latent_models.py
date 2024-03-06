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

def team_reb_pct_bayesian(per_game_fatten):
    #player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    #players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = {'oreb':[0,0],'dreb':[0,0]}


    last_lineup = {}
    cur_lineup = {}
    for team in teams:
        last_lineup[team] = {}
        cur_lineup[team] = {}


    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            #cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_oreb = team_game.at[0,'reboundsOffensive']
                a_team_oreb = team_game.at[1,'reboundsOffensive']
                h_team_dreb = team_game.at[0,'reboundsDefensive']
                a_team_dreb = team_game.at[1,'reboundsDefensive']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = team_game['team_id'].unique()[0]
            a_id = team_game['team_id'].unique()[1]

            # if ((date-last_game[h_id]).days == 1):
            #     h_b2b = True
            # else:
            #     h_b2b = False

            # if ((date-last_game[a_id]).days == 1):
            #     a_b2b = True
            # else:
            #     a_b2b = False
            
            # last_game[h_id] = date
            # last_game[a_id] = date

            # cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            # cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            # total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            # avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            # avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            # avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            # avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            # for x in [h_id, a_id]:
            #     cur_lineup[x] = {}
            #     expected_off_eff[x] = 0
            #     expected_def_eff[x] = 0
            #     for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
            #         cur_lineup[x][row['player_id']] = row['seconds']/total_sec

            #         expected_off_eff[x] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

            #         p_ids.append(player_map[row['player_id']])
            #         if (row['seconds'] != 0):
            #             p_sec_inv.append(((total_sec/5) / row['seconds']))
            #         else:
            #             p_sec_inv.append(100)
            #         if (x == h_id):
            #             opp_team_ids.append(a_id)
            #         else:
            #             opp_team_ids.append(h_id)
            #         if (x == h_id):
            #             obs.append(row['offensiveRating'] * h_team_eff/avg_h_off_player_eff - home_adv)
            #             if (h_b2b):
            #                 obs[-1] += b2b_pen
            #             if (a_b2b):
            #                 obs[-1] -= b2b_pen
            #         else:
            #             obs.append(row['offensiveRating'] * a_team_eff/avg_a_off_player_eff + home_adv)
            #             if (h_b2b):
            #                 obs[-1] -= b2b_pen
            #             if (a_b2b):
            #                 obs[-1] += b2b_pen
            #         off_def.append('off')

            #         expected_def_eff[x] += (row['seconds']/total_sec) * priors[2][player_map[row['player_id']]]

            #         p_ids.append(player_map[row['player_id']])
            #         if (row['seconds'] != 0):
            #             p_sec_inv.append(((total_sec/5) / row['seconds']))
            #         else:
            #             p_sec_inv.append(100)
            #         if (x == h_id):
            #             opp_team_ids.append(a_id)
            #         else:
            #             opp_team_ids.append(h_id)
            #         if (x == h_id):
            #             obs.append(row['defensiveRating'] * a_team_eff/avg_h_def_player_eff + home_adv)
            #             if (h_b2b):
            #                 obs[-1] -= b2b_pen
            #             if (a_b2b):
            #                 obs[-1] += b2b_pen
            #         else:
            #             obs.append(row['defensiveRating'] * h_team_eff/avg_a_def_player_eff - home_adv)
            #             if (h_b2b):
            #                 obs[-1] += b2b_pen
            #             if (a_b2b):
            #                 obs[-1] -= b2b_pen
            #         off_def.append('def')

    
            # cur_f = {'last_pred_h_eff':home_adv,'cur_pred_h_eff':home_adv,'last_pred_a_eff':-home_adv,'cur_pred_a_eff':-home_adv}
            # if (h_b2b):
            #     cur_f['last_pred_h_eff'] -= b2b_pen
            #     cur_f['cur_pred_h_eff'] -= b2b_pen
            #     cur_f['last_pred_a_eff'] += b2b_pen
            #     cur_f['cur_pred_a_eff'] += b2b_pen
            # if (a_b2b):
            #     cur_f['last_pred_a_eff'] -= b2b_pen
            #     cur_f['cur_pred_a_eff'] -= b2b_pen
            #     cur_f['last_pred_h_eff'] += b2b_pen
            #     cur_f['cur_pred_h_eff'] += b2b_pen
            # if (cur_season != "1996-97"):
            #     for x in [h_id,a_id]:
            #         for z in cur_lineup[x]:
            #             if (x == h_id):
            #                 cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
            #             else:
            #                 cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
            #         for z in last_lineup[x]:
            #             if (x == h_id):
            #                 cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            #             else:
            #                 cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            
            #     cur_f["actual_h_eff"] = h_team_eff
            #     cur_f["actual_a_eff"] = a_team_eff   
            #     features.append(cur_f)

            # last_lineup[h_id] = {}
            # for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
            #     last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            # last_lineup[a_id] = {}
            # for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
            #     last_lineup[a_id][row['player_id']] = row['seconds']/total_sec

            cur_f = {}
            if (priors[h_id]['dreb'][0] != 0 and priors[a_id]['dreb'][0] != 0):
                cur_f['pred_h_oreb_pct'] = ((priors[h_id]['oreb'][0] / (priors[h_id]['oreb'][0] + priors[h_id]['oreb'][1])) + (1 - (priors[a_id]['dreb'][0] / (priors[a_id]['dreb'][0] + priors[a_id]['dreb'][1])))) / 2
                cur_f['pred_a_oreb_pct'] = ((priors[a_id]['oreb'][0] / (priors[a_id]['oreb'][0] + priors[a_id]['oreb'][1])) + (1 - (priors[h_id]['dreb'][0] / (priors[h_id]['dreb'][0] + priors[h_id]['dreb'][1])))) / 2
            cur_f['actual_h_oreb_pct'] = team_game.at[0,'reboundsOffensive'] / (team_game.at[0,'reboundsOffensive'] + team_game.at[1,'reboundsDefensive'])
            cur_f['actual_a_oreb_pct'] = team_game.at[1,'reboundsOffensive'] / (team_game.at[1,'reboundsOffensive'] + team_game.at[0,'reboundsDefensive'])
            if (cur_season != "1996-97"):
                features.append(cur_f)

            #Update
            priors[h_id]['oreb'][0] += team_game.at[0,'reboundsOffensive']
            priors[h_id]['oreb'][1] += team_game.at[1,'reboundsDefensive']
            priors[h_id]['dreb'][0] += team_game.at[0,'reboundsDefensive']
            priors[h_id]['dreb'][1] += team_game.at[1,'reboundsOffensive']

            priors[a_id]['oreb'][0] += team_game.at[1,'reboundsOffensive']
            priors[a_id]['oreb'][1] += team_game.at[0,'reboundsDefensive']
            priors[a_id]['dreb'][0] += team_game.at[1,'reboundsDefensive']
            priors[a_id]['dreb'][1] += team_game.at[0,'reboundsOffensive']

            priors[h_id]['oreb'][0] *= per_game_fatten
            priors[h_id]['oreb'][1] *= per_game_fatten
            priors[h_id]['dreb'][0] *= per_game_fatten
            priors[h_id]['dreb'][1] *= per_game_fatten

            priors[a_id]['oreb'][0] *= per_game_fatten
            priors[a_id]['oreb'][1] *= per_game_fatten
            priors[a_id]['dreb'][0] *= per_game_fatten
            priors[a_id]['dreb'][1] *= per_game_fatten

        # opp_off_eff = []
        # opp_def_eff = []
        # for opp_id in opp_team_ids:
        #     opp_off_eff.append(expected_off_eff[opp_id])
        #     opp_def_eff.append(expected_def_eff[opp_id])

        # for i in range(len(p_ids)):
        #     if (p_sec_inv[i] > 25):
        #         continue
        #     if (off_def[i] == 'off'):
        #         priors[0][p_ids[i]] = (priors[0][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_def_eff[i])*priors[1][p_ids[i]]**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
        #         priors[1][p_ids[i]] = min(math.sqrt((priors[1][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)
        #     else:
        #         priors[2][p_ids[i]] = (priors[2][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_off_eff[i])*priors[3][p_ids[i]]**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
        #         priors[3][p_ids[i]] = min(math.sqrt((priors[3][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)

    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/base_oreb_pct_" + str(per_game_fatten) + ".csv", index=False)

    # plt.hist(priors[1])
    # plt.show()
    # plt.hist(priors[3])
    # plt.show()

def team_reb_pct_arma(weight):
    #player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    #players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = {'oreb':0.3,'dreb':0.7}



    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            #cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()


            try:
                #team_pace = team_game.at[0,"pace"]
                h_team_oreb = team_game.at[0,'reboundsOffensive']
                a_team_oreb = team_game.at[1,'reboundsOffensive']
                h_team_dreb = team_game.at[0,'reboundsDefensive']
                a_team_dreb = team_game.at[1,'reboundsDefensive']
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue

            h_id = team_game['team_id'].unique()[0]
            a_id = team_game['team_id'].unique()[1]

            # if ((date-last_game[h_id]).days == 1):
            #     h_b2b = True
            # else:
            #     h_b2b = False

            # if ((date-last_game[a_id]).days == 1):
            #     a_b2b = True
            # else:
            #     a_b2b = False
            
            # last_game[h_id] = date
            # last_game[a_id] = date

            # cur_game_h = cur_game.loc[cur_game['team_id']==h_id,]
            # cur_game_a = cur_game.loc[cur_game['team_id']==a_id,]

            # total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum()

            # avg_h_off_player_eff = (cur_game_h['seconds'] * cur_game_h['offensiveRating']).sum() / (cur_game_h['seconds'].sum())
            # avg_a_off_player_eff = (cur_game_a['seconds'] * cur_game_a['offensiveRating']).sum() / (cur_game_a['seconds'].sum())
            # avg_h_def_player_eff = (cur_game_h['seconds'] * cur_game_h['defensiveRating']).sum() / (cur_game_h['seconds'].sum())
            # avg_a_def_player_eff = (cur_game_a['seconds'] * cur_game_a['defensiveRating']).sum() / (cur_game_a['seconds'].sum())

            # for x in [h_id, a_id]:
            #     cur_lineup[x] = {}
            #     expected_off_eff[x] = 0
            #     expected_def_eff[x] = 0
            #     for index, row in cur_game.loc[cur_game['team_id'] == x, ].iterrows():
            #         cur_lineup[x][row['player_id']] = row['seconds']/total_sec

            #         expected_off_eff[x] += (row['seconds']/total_sec) * priors[0][player_map[row['player_id']]]

            #         p_ids.append(player_map[row['player_id']])
            #         if (row['seconds'] != 0):
            #             p_sec_inv.append(((total_sec/5) / row['seconds']))
            #         else:
            #             p_sec_inv.append(100)
            #         if (x == h_id):
            #             opp_team_ids.append(a_id)
            #         else:
            #             opp_team_ids.append(h_id)
            #         if (x == h_id):
            #             obs.append(row['offensiveRating'] * h_team_eff/avg_h_off_player_eff - home_adv)
            #             if (h_b2b):
            #                 obs[-1] += b2b_pen
            #             if (a_b2b):
            #                 obs[-1] -= b2b_pen
            #         else:
            #             obs.append(row['offensiveRating'] * a_team_eff/avg_a_off_player_eff + home_adv)
            #             if (h_b2b):
            #                 obs[-1] -= b2b_pen
            #             if (a_b2b):
            #                 obs[-1] += b2b_pen
            #         off_def.append('off')

            #         expected_def_eff[x] += (row['seconds']/total_sec) * priors[2][player_map[row['player_id']]]

            #         p_ids.append(player_map[row['player_id']])
            #         if (row['seconds'] != 0):
            #             p_sec_inv.append(((total_sec/5) / row['seconds']))
            #         else:
            #             p_sec_inv.append(100)
            #         if (x == h_id):
            #             opp_team_ids.append(a_id)
            #         else:
            #             opp_team_ids.append(h_id)
            #         if (x == h_id):
            #             obs.append(row['defensiveRating'] * a_team_eff/avg_h_def_player_eff + home_adv)
            #             if (h_b2b):
            #                 obs[-1] -= b2b_pen
            #             if (a_b2b):
            #                 obs[-1] += b2b_pen
            #         else:
            #             obs.append(row['defensiveRating'] * h_team_eff/avg_a_def_player_eff - home_adv)
            #             if (h_b2b):
            #                 obs[-1] += b2b_pen
            #             if (a_b2b):
            #                 obs[-1] -= b2b_pen
            #         off_def.append('def')

    
            # cur_f = {'last_pred_h_eff':home_adv,'cur_pred_h_eff':home_adv,'last_pred_a_eff':-home_adv,'cur_pred_a_eff':-home_adv}
            # if (h_b2b):
            #     cur_f['last_pred_h_eff'] -= b2b_pen
            #     cur_f['cur_pred_h_eff'] -= b2b_pen
            #     cur_f['last_pred_a_eff'] += b2b_pen
            #     cur_f['cur_pred_a_eff'] += b2b_pen
            # if (a_b2b):
            #     cur_f['last_pred_a_eff'] -= b2b_pen
            #     cur_f['cur_pred_a_eff'] -= b2b_pen
            #     cur_f['last_pred_h_eff'] += b2b_pen
            #     cur_f['cur_pred_h_eff'] += b2b_pen
            # if (cur_season != "1996-97"):
            #     for x in [h_id,a_id]:
            #         for z in cur_lineup[x]:
            #             if (x == h_id):
            #                 cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
            #             else:
            #                 cur_f['cur_pred_a_eff'] += cur_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['cur_pred_h_eff'] += cur_lineup[x][z] * priors[2][player_map[z]]
            #         for z in last_lineup[x]:
            #             if (x == h_id):
            #                 cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            #             else:
            #                 cur_f['last_pred_a_eff'] += last_lineup[x][z] * priors[0][player_map[z]]
            #                 cur_f['last_pred_h_eff'] += last_lineup[x][z] * priors[2][player_map[z]]
            
            #     cur_f["actual_h_eff"] = h_team_eff
            #     cur_f["actual_a_eff"] = a_team_eff   
            #     features.append(cur_f)

            # last_lineup[h_id] = {}
            # for index, row in cur_game.loc[cur_game['team_id'] == h_id, ].iterrows():
            #     last_lineup[h_id][row['player_id']] = row['seconds']/total_sec
            # last_lineup[a_id] = {}
            # for index, row in cur_game.loc[cur_game['team_id'] == a_id, ].iterrows():
            #     last_lineup[a_id][row['player_id']] = row['seconds']/total_sec

            cur_f = {}
            cur_f['pred_h_oreb_pct'] = (priors[h_id]['oreb'] + (1-priors[a_id]['dreb'])) / 2
            cur_f['pred_a_oreb_pct'] = (priors[a_id]['oreb'] + (1-priors[h_id]['dreb'])) / 2
            cur_f['actual_h_oreb_pct'] = team_game.at[0,'reboundsOffensive'] / (team_game.at[0,'reboundsOffensive'] + team_game.at[1,'reboundsDefensive'])
            cur_f['actual_a_oreb_pct'] = team_game.at[1,'reboundsOffensive'] / (team_game.at[1,'reboundsOffensive'] + team_game.at[0,'reboundsDefensive'])
            if (cur_season != "1996-97"):
                features.append(cur_f)

            
            #Update
            priors[h_id]['oreb'] += (cur_f['actual_h_oreb_pct'] - cur_f['pred_h_oreb_pct']) * weight
            priors[a_id]['oreb'] += (cur_f['actual_a_oreb_pct'] - cur_f['pred_a_oreb_pct']) * weight
            priors[h_id]['dreb'] += ((1-cur_f['actual_a_oreb_pct']) - (1-cur_f['pred_a_oreb_pct'])) * weight
            priors[a_id]['dreb'] += ((1-cur_f['actual_h_oreb_pct']) - (1-cur_f['pred_h_oreb_pct'])) * weight


        # opp_off_eff = []
        # opp_def_eff = []
        # for opp_id in opp_team_ids:
        #     opp_off_eff.append(expected_off_eff[opp_id])
        #     opp_def_eff.append(expected_def_eff[opp_id])

        # for i in range(len(p_ids)):
        #     if (p_sec_inv[i] > 25):
        #         continue
        #     if (off_def[i] == 'off'):
        #         priors[0][p_ids[i]] = (priors[0][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_def_eff[i])*priors[1][p_ids[i]]**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
        #         priors[1][p_ids[i]] = min(math.sqrt((priors[1][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[1][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)
        #     else:
        #         priors[2][p_ids[i]] = (priors[2][p_ids[i]]*(p_sec_inv[i]*obs_sigma)**2 + (obs[i] - opp_off_eff[i])*priors[3][p_ids[i]]**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)
        #         priors[3][p_ids[i]] = min(math.sqrt((priors[3][p_ids[i]]**2*(p_sec_inv[i]*obs_sigma)**2) / (priors[3][p_ids[i]]**2 + (p_sec_inv[i]*obs_sigma)**2)) * per_game_fatten, max_sigma)

    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()
    
    print (games)
    fdf = pd.DataFrame(features)
    print  (fdf)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/arma_oreb_pct_" + str(weight) + ".csv", index=False)

    # plt.hist(priors[1])
    # plt.show()
    # plt.hist(priors[3])
    # plt.show()

#0.2 is optimal
def usg_pct_arma(weight):
    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
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

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = 0.2


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

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                cur_f['pred_usg'] = priors[row['player_id']]
                cur_f['actual_usg'] = row['usagePercentage']
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                time_played = row['seconds'] / total_sec
                priors[row['player_id']] += (row['usagePercentage'] - priors[row['player_id']]) * weight * time_played

        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/arma_time_usg_pct_" + str(weight) + ".csv", index=False)

    # plt.hist(priors[1])
    # plt.show()
    # plt.hist(priors[3])
    # plt.show()

def usg_pct_bayesian(per_game_fatten, obs_sigma):
    player_bs = pd.read_csv("./database/advanced_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0.2,obs_sigma]


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

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                cur_f['pred_usg'] = priors[row['player_id']][0]
                cur_f['actual_usg'] = row['usagePercentage']
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                if (row['seconds'] > 0):
                    p_sec_inv = total_sec / row['seconds']
                    priors[row['player_id']][0] = (priors[row['player_id']][0]*(p_sec_inv*obs_sigma)**2 + (row['usagePercentage'])*priors[row['player_id']][1]**2) / (priors[row['player_id']][1]**2 + (p_sec_inv*obs_sigma)**2)
                    priors[row['player_id']][1] = math.sqrt((priors[row['player_id']][1]**2*(p_sec_inv*obs_sigma)**2) / (priors[row['player_id']][1]**2 + (p_sec_inv*obs_sigma)**2)) * per_game_fatten

        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_usg_pct_"+ str(obs_sigma) +"_"+ str(per_game_fatten) + ".csv", index=False)

    # plt.hist(priors[1])
    # plt.show()
    # plt.hist(priors[3])
    # plt.show()

#The following are supposed to be baseline metrics for predicting
#We will also feed the model more recent form alongside these

#optimal is 0.997
def player_3pt_pct_offense(per_game_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    abs_error = []
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
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                    cur_f['pred_3pt_pct'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                if (row['threePointersAttempted'] > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                    cur_f['actual_3pt_pct'] = row['threePointersMade'] / row['threePointersAttempted']
                    abs_error.append(abs(cur_f['actual_3pt_pct'] - cur_f['pred_3pt_pct']))
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                priors[row['player_id']][0] += row['threePointersMade']
                priors[row['player_id']][1] += row['threePointersAttempted'] - row['threePointersMade']

                priors[row['player_id']][0] *= per_game_fatten_beta
                priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_3pt_pct_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.99
def player_2pt_pct_offense(per_game_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    abs_error = []
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
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                    cur_f['pred_2pt_pct'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                if (row['fieldGoalsAttempted'] - row['threePointersAttempted'] > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                    cur_f['actual_2pt_pct'] = (row['fieldGoalsMade'] - row['threePointersMade']) / (row['fieldGoalsAttempted'] - row['threePointersAttempted'])
                    abs_error.append(abs(cur_f['actual_2pt_pct'] - cur_f['pred_2pt_pct']))
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                priors[row['player_id']][0] += row['fieldGoalsMade'] - row['threePointersMade']
                priors[row['player_id']][1] += (row['fieldGoalsAttempted'] - row['threePointersAttempted']) - (row['fieldGoalsMade'] - row['threePointersMade'])

                priors[row['player_id']][0] *= per_game_fatten_beta
                priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_2pt_pct_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.98
def player_ft_pct(per_game_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    abs_error = []
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
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                    cur_f['pred_ft_pct'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                if (row['freeThrowsAttempted'] > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                    cur_f['actual_ft_pct'] = row['freeThrowsMade'] / row['freeThrowsAttempted']
                    abs_error.append(abs(cur_f['actual_ft_pct'] - cur_f['pred_ft_pct'])*row['freeThrowsAttempted'])
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                priors[row['player_id']][0] += row['freeThrowsMade']
                priors[row['player_id']][1] += row['freeThrowsAttempted'] - row['freeThrowsMade']

                priors[row['player_id']][0] *= per_game_fatten_beta
                priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_ft_pct_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.98
def player_to_pct_offense(per_game_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    abs_error = []
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
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for index, row in cur_game.iterrows():
                end_poss_ct = row['fieldGoalsAttempted'] + row['freeThrowsAttempted']*0.44 + row['assists'] + row['turnovers']
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                    cur_f['pred_to_pct'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                if (end_poss_ct > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                    cur_f['actual_to_pct'] = row['turnovers'] / end_poss_ct
                    abs_error.append(abs(cur_f['actual_to_pct'] - cur_f['pred_to_pct'])*end_poss_ct)
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                priors[row['player_id']][0] += row['turnovers']
                priors[row['player_id']][1] += end_poss_ct - row['turnovers']

                priors[row['player_id']][0] *= per_game_fatten_beta
                priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_to_pct_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))

#Not the traditional free throw rate: ft attempted / (ft attempted + fg attempted)
#optimal is 0.97
def player_ftr_offense(per_game_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    abs_error = []
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
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for index, row in cur_game.iterrows():
                cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                    cur_f['pred_ftr'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                if (row['fieldGoalsAttempted'] + row['freeThrowsAttempted'] > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                    cur_f['actual_ftr'] = row['freeThrowsAttempted'] / (row['freeThrowsAttempted'] + row['fieldGoalsAttempted'])
                    abs_error.append(abs(cur_f['actual_ftr'] - cur_f['pred_ftr'])*(row['freeThrowsAttempted'] + row['fieldGoalsAttempted']))
                if (cur_season != "1996-97"):
                    features.append(cur_f)

                #Update
                priors[row['player_id']][0] += row['freeThrowsAttempted']
                priors[row['player_id']][1] += row['fieldGoalsAttempted']

                priors[row['player_id']][0] *= per_game_fatten_beta
                priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_ftr_" + str(per_game_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.995, 1
def team_3pt_pct_defense(per_game_fatten_beta, per_season_fatten_beta):
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            
            if (last_game_season != cur_season and last_game_season != -1):
                for team in teams:
                    priors[team][0] *= per_season_fatten_beta
                    priors[team][1] *= per_season_fatten_beta
            last_game_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5
            
            cur_f = {}
            if (priors[h_id][1] + priors[h_id][0] > 0):
                cur_f['pred_h_def_3pt_pct'] = priors[h_id][0] / (priors[h_id][1] + priors[h_id][0])
            if (priors[a_id][1] + priors[a_id][0] > 0):
                cur_f['pred_a_def_3pt_pct'] = priors[a_id][0] / (priors[a_id][1] + priors[a_id][0])
            if (team_game.at[1,'threePointersAttempted'] > 0 and (priors[h_id][1] + priors[h_id][0] > 0)):
                cur_f['actual_h_def_3pt_pct'] = team_game.at[1,'threePointersMade'] / team_game.at[1,'threePointersAttempted']
                abs_error.append(abs(cur_f['actual_h_def_3pt_pct'] - cur_f['pred_h_def_3pt_pct'])*(team_game.at[1,'threePointersAttempted']))
            if (team_game.at[0,'threePointersAttempted'] > 0 and (priors[a_id][1] + priors[a_id][0] > 0)):
                cur_f['actual_a_def_3pt_pct'] = team_game.at[0,'threePointersMade'] / team_game.at[0,'threePointersAttempted']
                abs_error.append(abs(cur_f['actual_a_def_3pt_pct'] - cur_f['pred_a_def_3pt_pct'])*(team_game.at[0,'threePointersAttempted']))
            if (cur_season != "1996-97"):
                features.append(cur_f)

            #Update
            priors[h_id][0] += team_game.at[1,'threePointersMade']
            priors[h_id][1] += team_game.at[1,'threePointersAttempted'] - team_game.at[1,'threePointersMade']

            priors[h_id][0] *= per_game_fatten_beta
            priors[h_id][1] *= per_game_fatten_beta

            priors[a_id][0] += team_game.at[0,'threePointersMade']
            priors[a_id][1] += team_game.at[0,'threePointersAttempted'] - team_game.at[0,'threePointersMade']

            priors[a_id][0] *= per_game_fatten_beta
            priors[a_id][1] *= per_game_fatten_beta
        
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()

    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/bayes_def_3pt_pct_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.974, 0.5
def team_2pt_pct_defense(per_game_fatten_beta, per_season_fatten_beta):
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            
            if (last_game_season != cur_season and last_game_season != -1):
                for team in teams:
                    priors[team][0] *= per_season_fatten_beta
                    priors[team][1] *= per_season_fatten_beta
            last_game_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5
            
            cur_f = {}
            if (priors[h_id][1] + priors[h_id][0] > 0):
                cur_f['pred_h_def_2pt_pct'] = priors[h_id][0] / (priors[h_id][1] + priors[h_id][0])
            if (priors[a_id][1] + priors[a_id][0] > 0):
                cur_f['pred_a_def_2pt_pct'] = priors[a_id][0] / (priors[a_id][1] + priors[a_id][0])
            if (team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted'] > 0 and (priors[h_id][1] + priors[h_id][0] > 0)):
                cur_f['actual_h_def_2pt_pct'] = (team_game.at[1,'fieldGoalsMade'] - team_game.at[1,'threePointersMade']) / (team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted'])
                abs_error.append(abs(cur_f['actual_h_def_2pt_pct'] - cur_f['pred_h_def_2pt_pct'])*((team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted'])))
            if (team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted'] > 0 and (priors[a_id][1] + priors[a_id][0] > 0)):
                cur_f['actual_a_def_2pt_pct'] = (team_game.at[0,'fieldGoalsMade'] - team_game.at[0,'threePointersMade']) / (team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted'])
                abs_error.append(abs(cur_f['actual_a_def_2pt_pct'] - cur_f['pred_a_def_2pt_pct'])*((team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted'])))
            if (cur_season != "1996-97"):
                features.append(cur_f)

            #Update
            priors[h_id][0] += team_game.at[1,'fieldGoalsMade'] - team_game.at[1,'threePointersMade']
            priors[h_id][1] += (team_game.at[1,'fieldGoalsAttempted'] - team_game.at[1,'threePointersAttempted']) - (team_game.at[1,'fieldGoalsMade'] - team_game.at[1,'threePointersMade'])

            priors[h_id][0] *= per_game_fatten_beta
            priors[h_id][1] *= per_game_fatten_beta

            priors[a_id][0] += team_game.at[0,'fieldGoalsMade'] - team_game.at[0,'threePointersMade']
            priors[a_id][1] += (team_game.at[0,'fieldGoalsAttempted'] - team_game.at[0,'threePointersAttempted']) - (team_game.at[0,'fieldGoalsMade'] - team_game.at[0,'threePointersMade'])

            priors[a_id][0] *= per_game_fatten_beta
            priors[a_id][1] *= per_game_fatten_beta
        
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()

    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/bayes_def_2pt_pct_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.96, 0.8
def team_to_pct_defense(per_game_fatten_beta, per_season_fatten_beta):
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            
            if (last_game_season != cur_season and last_game_season != -1):
                for team in teams:
                    priors[team][0] *= per_season_fatten_beta
                    priors[team][1] *= per_season_fatten_beta
            last_game_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5
            
            cur_f = {}
            h_def_end_poss_ct = team_game.at[1,'fieldGoalsAttempted'] + team_game.at[1,'freeThrowsAttempted']*0.44 + team_game.at[1,'turnovers']
            a_def_end_poss_ct = team_game.at[0,'fieldGoalsAttempted'] + team_game.at[0,'freeThrowsAttempted']*0.44 + team_game.at[0,'turnovers']
            if (priors[h_id][1] + priors[h_id][0] > 0):
                cur_f['pred_h_def_to_pct'] = priors[h_id][0] / (priors[h_id][1] + priors[h_id][0])
            if (priors[a_id][1] + priors[a_id][0] > 0):
                cur_f['pred_a_def_to_pct'] = priors[a_id][0] / (priors[a_id][1] + priors[a_id][0])
            if ((priors[h_id][1] + priors[h_id][0] > 0)):
                cur_f['actual_h_def_to_pct'] = team_game.at[1,'turnovers'] / h_def_end_poss_ct
                abs_error.append(abs(cur_f['actual_h_def_to_pct'] - cur_f['pred_h_def_to_pct']))
            if ((priors[a_id][1] + priors[a_id][0] > 0)):
                cur_f['actual_a_def_to_pct'] = team_game.at[0,'turnovers'] / a_def_end_poss_ct
                abs_error.append(abs(cur_f['actual_a_def_to_pct'] - cur_f['pred_a_def_to_pct']))
            if (cur_season != "1996-97"):
                features.append(cur_f)

            #Update
            priors[h_id][0] += team_game.at[1,'turnovers']
            priors[h_id][1] += h_def_end_poss_ct - team_game.at[1,'turnovers']

            priors[h_id][0] *= per_game_fatten_beta
            priors[h_id][1] *= per_game_fatten_beta

            priors[a_id][0] += team_game.at[0,'turnovers']
            priors[a_id][1] += a_def_end_poss_ct - team_game.at[0,'turnovers']

            priors[a_id][0] *= per_game_fatten_beta
            priors[a_id][1] *= per_game_fatten_beta
        
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()

    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/bayes_def_to_pct_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.96, 0.7
def team_ftr_defense(per_game_fatten_beta, per_season_fatten_beta):
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (teams):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
            
            if (last_game_season != cur_season and last_game_season != -1):
                for team in teams:
                    priors[team][0] *= per_season_fatten_beta
                    priors[team][1] *= per_season_fatten_beta
            last_game_season = cur_season

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5
            
            cur_f = {}
            if (priors[h_id][1] + priors[h_id][0] > 0):
                cur_f['pred_h_def_ftr'] = priors[h_id][0] / (priors[h_id][1] + priors[h_id][0])
            if (priors[a_id][1] + priors[a_id][0] > 0):
                cur_f['pred_a_def_ftr'] = priors[a_id][0] / (priors[a_id][1] + priors[a_id][0])
            if (team_game.at[1,'fieldGoalsAttempted'] + team_game.at[1,'freeThrowsAttempted'] > 0 and (priors[h_id][1] + priors[h_id][0] > 0)):
                cur_f['actual_h_def_ftr'] = team_game.at[1,'freeThrowsAttempted'] / (team_game.at[1,'fieldGoalsAttempted'] + team_game.at[1,'freeThrowsAttempted'])
                abs_error.append(abs(cur_f['actual_h_def_ftr'] - cur_f['pred_h_def_ftr']))
            if (team_game.at[0,'fieldGoalsAttempted'] + team_game.at[0,'freeThrowsAttempted'] > 0 and (priors[a_id][1] + priors[a_id][0] > 0)):
                cur_f['actual_a_def_ftr'] = team_game.at[0,'freeThrowsAttempted'] / (team_game.at[0,'fieldGoalsAttempted']+team_game.at[0,'freeThrowsAttempted'])
                abs_error.append(abs(cur_f['actual_a_def_ftr'] - cur_f['pred_a_def_ftr']))
            if (cur_season != "1996-97"):
                features.append(cur_f)

            #Update
            priors[h_id][0] += team_game.at[1,'freeThrowsAttempted']
            priors[h_id][1] += team_game.at[1,'fieldGoalsAttempted']

            priors[h_id][0] *= per_game_fatten_beta
            priors[h_id][1] *= per_game_fatten_beta

            priors[a_id][0] += team_game.at[0,'freeThrowsAttempted']
            priors[a_id][1] += team_game.at[0,'fieldGoalsAttempted']

            priors[a_id][0] *= per_game_fatten_beta
            priors[a_id][1] *= per_game_fatten_beta
        
    seasons = ""
    for yr in range(1997, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index()

    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/latent/bayes_def_ftr_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.85, 0.2
def usg_3pt(per_game_fatten_beta, per_season_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()

            if (last_game_season != cur_season and last_game_season != -1):
                for player in players:
                    priors[player][0] *= per_season_fatten_beta
                    priors[player][1] *= per_season_fatten_beta
            last_game_season = cur_season
            

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for x in [h_id,a_id]:
                cur_side = cur_game.loc[cur_game['team_id']==x,].reset_index(drop=True)
                total_attempted = cur_side['threePointersAttempted'].sum()
                for index, row in cur_side.iterrows():
                    cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                    if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                        cur_f['pred_3pt_usg'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                    if (total_attempted > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                        cur_f['actual_3pt_usg'] = (row['threePointersAttempted']) / (total_attempted)
                        abs_error.append(abs(cur_f['actual_3pt_usg'] - cur_f['pred_3pt_usg']))
                    if (cur_season != "1996-97"):
                        features.append(cur_f)

                    #Update
                    priors[row['player_id']][0] += row['threePointersAttempted']
                    priors[row['player_id']][1] += total_attempted - row['threePointersAttempted']

                    priors[row['player_id']][0] *= per_game_fatten_beta
                    priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_3pt_usg_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.85, 0.2
def usg_2pt(per_game_fatten_beta, per_season_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()

            if (last_game_season != cur_season and last_game_season != -1):
                for player in players:
                    priors[player][0] *= per_season_fatten_beta
                    priors[player][1] *= per_season_fatten_beta
            last_game_season = cur_season
            

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for x in [h_id,a_id]:
                cur_side = cur_game.loc[cur_game['team_id']==x,].reset_index(drop=True)
                total_attempted = cur_side['fieldGoalsAttempted'].sum() - cur_side['threePointersAttempted'].sum()
                for index, row in cur_side.iterrows():
                    cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                    if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                        cur_f['pred_2pt_usg'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                    if (total_attempted > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                        cur_f['actual_2pt_usg'] = (row['fieldGoalsAttempted'] - row['threePointersAttempted']) / (total_attempted)
                        abs_error.append(abs(cur_f['actual_2pt_usg'] - cur_f['pred_2pt_usg']))
                    if (cur_season != "1996-97"):
                        features.append(cur_f)

                    #Update
                    priors[row['player_id']][0] += row['fieldGoalsAttempted'] - row['threePointersAttempted']
                    priors[row['player_id']][1] += total_attempted - (row['fieldGoalsAttempted'] - row['threePointersAttempted'])

                    priors[row['player_id']][0] *= per_game_fatten_beta
                    priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_2pt_usg_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#optimal is 0.85, 0.2
def usg_fg(per_game_fatten_beta, per_season_fatten_beta):
    player_bs = pd.read_csv("./database/traditional_boxscores_players.csv")
    team_bs = pd.read_csv("./database/traditional_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2003):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    teams = games['h_team_id'].unique()
    players = player_bs['player_id'].unique()
    games['game_date'] = pd.to_datetime(games['game_date'])

    #player_bs['seconds'] = player_bs['minutes'].dropna().transform(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))

    features = []

    priors = {}
    for i in (players):
        priors[i] = [0,0]

    last_game_season = -1
    abs_error = []
    for date in tqdm(games['game_date'].unique()):
        game_ids = games.loc[games['game_date']==date,]['game_id'].unique()       

        for gid in game_ids:
            cur_game = player_bs.loc[player_bs["game_id"] == gid,].dropna().reset_index()
            cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
            team_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()

            if (last_game_season != cur_season and last_game_season != -1):
                for player in players:
                    priors[player][0] *= per_season_fatten_beta
                    priors[player][1] *= per_season_fatten_beta
            last_game_season = cur_season
            

            try:
                #pace_test = cur_game.at[0,"pace"]
                h_id = team_game['team_id'].unique()[0]
                a_id = team_game['team_id'].unique()[1]
            except:
                if (cur_season != "1996-97"):
                    cur_f = {}
                    features.append(cur_f)
                continue
            
            #total_sec = cur_game.loc[cur_game['team_id'] == h_id, ]['seconds'].sum() / 5

            for x in [h_id,a_id]:
                cur_side = cur_game.loc[cur_game['team_id']==x,].reset_index(drop=True)
                total_attempted = cur_side['fieldGoalsAttempted'].sum()
                for index, row in cur_side.iterrows():
                    cur_f = {'game_id':row['game_id'],'player_id':row['player_id'],'team_id':row['team_id'],'minutes':row['minutes']}
                    if (priors[row['player_id']][1] + priors[row['player_id']][0] > 0):
                        cur_f['pred_fg_usg'] = priors[row['player_id']][0] / (priors[row['player_id']][1] + priors[row['player_id']][0])
                    if (total_attempted > 0 and (priors[row['player_id']][1] + priors[row['player_id']][0] > 0)):
                        cur_f['actual_fg_usg'] = (row['fieldGoalsAttempted']) / (total_attempted)
                        abs_error.append(abs(cur_f['actual_fg_usg'] - cur_f['pred_fg_usg']))
                    if (cur_season != "1996-97"):
                        features.append(cur_f)

                    #Update
                    priors[row['player_id']][0] += row['fieldGoalsAttempted']
                    priors[row['player_id']][1] += total_attempted - (row['fieldGoalsAttempted'])

                    priors[row['player_id']][0] *= per_game_fatten_beta
                    priors[row['player_id']][1] *= per_game_fatten_beta
        
    fdf = pd.DataFrame(features)

    fdf.to_csv("./predictions/latent/bayes_fg_usg_" + str(per_game_fatten_beta) + "_" + str(per_season_fatten_beta) + ".csv", index=False)
    print (np.mean(abs_error))
    print (per_game_fatten_beta)
    print (per_season_fatten_beta)
    return (np.mean(abs_error))

#for x in [0.8,0.825,0.85,0.875,0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]:
#for x in [0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,1]:
#for x in [0.97,0.972,0.974,0.976,0.978,0.98,0.982,0.984,0.986,0.988,0.99,0.992,0.994,0.996,0.998,1]:
for x in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    usg_fg(0.85,x)