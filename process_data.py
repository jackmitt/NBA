import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import datetime

db_path = "./database/"
i_path = "./intermediates/"
p_path = "./processed/"

#Each stat feature for the model is generated in some way based on all games prior
#This class contains everything for those operations
#stat_list is a list of all stats from the boxscore to be retrieved after every game and used in some way to generate features for prediction for future games
#container is what holds the stats before processing
class stat_king:
    def __init__(self, stat_list):
        self.stat_list = stat_list
        self.container = {}

        teams = pd.read_csv(db_path+"teams.csv")["id"].to_list()
        for team in teams:
            self.container[team] = {"games_played":0}
            for stat in stat_list:
                self.container[team][stat] = []
                self.container[team]["opp_"+stat] = []
                self.container[team]["prior_"+stat] = -1
                self.container[team]["prior_opp_"+stat] = -1
    
    #subset is the df of the pair of rows from the boxscore for a game_id
    def update(self, subset):
        subset = subset.reset_index(drop=True)
        mirror = [1,0]
        for i in range(2):
            self.container[subset.at[i,"team_id"]]["games_played"] += 1
            for stat in self.stat_list:
                self.container[subset.at[i,"team_id"]][stat].append(subset.at[i,stat])
                self.container[subset.at[i,"team_id"]]["opp_"+stat].append(subset.at[mirror[i],stat])
    
    def new_season(self):
        teams = pd.read_csv(db_path+"teams.csv")["id"].to_list()
        for team in teams:
            self.container[team]["games_played"] = 0
            for stat in self.stat_list:
                self.container[team]["prior_"+stat] = np.average(self.container[team][stat])
                self.container[team]["prior_opp_"+stat] = np.average(self.container[team]["opp_"+stat])
                self.container[team][stat] = []
                self.container[team]["opp_"+stat] = []

    #teams is a tuple of team_ids for a game with the first being the home team
    #returns tuple of sub_containers for a team
    def get_stats(self, teams):
        return ((self.container[teams[0]], self.container[teams[1]]))



def train_test_val_split():
    pass

#response_type is based on whether the response for the model is the points of each team (team) or win/loss, spread, total (game)
def beginning_baseline():
    t_trad = pd.read_csv(db_path+"traditional_boxscores_teams.csv")

    games = pd.read_csv(db_path+"games.csv")
    #exclude first two seasons bc of missing data
    seasons = ""
    for yr in range(1998, 2023):
        seasons += str(yr) + "-" + str(yr+1)[2:4]+"|"
    games = games[games["season"].str.contains(seasons[:-1])]
    games = games[games["game_type"].str.contains("Regular Season|Playoffs")]

    sk = stat_king(["points"])

    features = {"h_avg_points":[],"h_avg_opp_points":[],"a_avg_points":[],"a_avg_opp_points":[]}
    responses = {"h_points":[],"a_points":[]}

    drop_games = []
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(1998, 2023)]
    h_m = ["h_","a_"]
    for season in seasons:
        season_games = games.loc[games["season"]==season,]
        for game_id in tqdm(season_games["game_id"].unique()):
            game_subset = t_trad.loc[t_trad["game_id"]==game_id, ]
            if (len(game_subset.index) != 2):
                drop_games.append(game_id)
                continue

            team_stats = sk.get_stats((game_subset.at[game_subset.index[0],"team_id"], game_subset.at[game_subset.index[1],"team_id"]))

            #no predictions for first 5 in first season since we need prior season's
            if (season == "1998-99" and (team_stats[0]["games_played"] < 5 or team_stats[1]["games_played"] < 5)):
                for key in features:
                    features[key].append(np.nan)
                for key in responses:
                    responses[key].append(np.nan)
                sk.update(game_subset)
                continue

            
            for i in range(2):
                if (team_stats[i]["games_played"] == 0):
                    features[h_m[i]+"avg_points"].append(team_stats[i]["prior_points"])
                    features[h_m[i]+"avg_opp_points"].append(team_stats[i]["prior_opp_points"])
                elif (team_stats[i]["games_played"] < 5):
                    features[h_m[i]+"avg_points"].append((5-team_stats[i]["games_played"])/5 * team_stats[i]["prior_points"] + team_stats[i]["games_played"]/5 * np.average(team_stats[i]["points"]))
                    features[h_m[i]+"avg_opp_points"].append((5-team_stats[i]["games_played"])/5 * team_stats[i]["prior_opp_points"] + team_stats[i]["games_played"]/5 * np.average(team_stats[i]["opp_points"]))
                else:
                    features[h_m[i]+"avg_points"].append(np.average(team_stats[i]["points"]))
                    features[h_m[i]+"avg_opp_points"].append(np.average(team_stats[i]["opp_points"]))
                responses[h_m[i]+"points"].append(game_subset.at[game_subset.index[i],"points"])            

            sk.update(game_subset)
        sk.new_season()
    games = games.set_index("game_id").drop(drop_games)
    for key in features:
        games[key] = features[key]
    for key in responses:
        games[key] = responses[key]
    games.to_csv(p_path+"beginning_baseline.csv")


