import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import pickle
import math

#bad model just to make predictions so that I can work on other aspects of the process
def placeholder():
    team_bs = pd.read_csv("./database/advanced_boxscores_teams.csv")
    games = pd.read_csv("./database/games.csv")
    seasons = ""
    for yr in range(1996, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]
    game_ids = games["game_id"].to_list()

    arma = {'pace':{},'rtg':{}}


    features = []

    for team in games["h_team_id"].unique():
        arma['rtg'][team] = [50,50]
        arma['pace'][team] = 47.5
    
    for gid in tqdm(game_ids):
        cur_game = team_bs.loc[team_bs["game_id"] == gid,].reset_index()
        cur_season = games.loc[games["game_id"] == gid, ]["season"].to_list()[0]
        pri_season = games["season"].unique()[games["season"].unique().tolist().index(cur_season)-1]

        try:
            h_rtg = cur_game.at[0,"offensiveRating"]
            a_rtg = cur_game.at[1,"offensiveRating"]
            pace = cur_game.at[0,"pace"]
        except:
            if (cur_season != "1996-97"):
                cur_f = {}
                features.append(cur_f)
            continue

        cur_f = {}
        if (cur_season != "1996-97"):
            cur_f['pace_pred'] = arma['pace'][cur_game.at[0,"team_id"]] + arma['pace'][cur_game.at[1,"team_id"]]
            cur_f['h_rtg_pred'] = arma['rtg'][cur_game.at[0,"team_id"]][0] + arma['rtg'][cur_game.at[1,"team_id"]][1]
            cur_f['a_rtg_pred'] = arma['rtg'][cur_game.at[0,"team_id"]][1] + arma['rtg'][cur_game.at[1,"team_id"]][0]
        
            cur_f['actual_pace'] = pace
            cur_f["actual_h"] = h_rtg 
            cur_f["actual_a"] = a_rtg   
            features.append(cur_f)

        #Updating below

        error_term_pace = pace - arma['pace'][cur_game.at[0,"team_id"]] - arma['pace'][cur_game.at[1,"team_id"]]
        arma['pace'][cur_game.at[0,"team_id"]] += error_term_pace*0.10
        arma['pace'][cur_game.at[1,"team_id"]] += error_term_pace*0.10

        error_term_h = h_rtg - arma['rtg'][cur_game.at[0,"team_id"]][0] - arma['rtg'][cur_game.at[1,"team_id"]][1]
        error_term_a = a_rtg - arma['rtg'][cur_game.at[1,"team_id"]][0] - arma['rtg'][cur_game.at[0,"team_id"]][1]
        
        arma['rtg'][cur_game.at[0,"team_id"]][0] += error_term_h*0.05
        arma['rtg'][cur_game.at[1,"team_id"]][1] += error_term_h*0.05
        arma['rtg'][cur_game.at[1,"team_id"]][0] += error_term_a*0.05
        arma['rtg'][cur_game.at[0,"team_id"]][1] += error_term_a*0.05
    
    seasons = ""
    for yr in range(1997, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])].reset_index(drop=True)
    

    fdf = pd.DataFrame(features)
    games = pd.concat([games, fdf], axis=1)
    games.to_csv("./predictions/betting/placeholder.csv", index=False)

placeholder()