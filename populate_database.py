from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscoremiscv3
from nba_api.stats.endpoints import boxscorescoringv3
from nba_api.stats.endpoints import boxscoreusagev3
from nba_api.stats.endpoints import boxscorefourfactorsv3
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from os.path import exists
import os
import time

db_path = "./database/"

def teams_table():
    pd.DataFrame(teams.get_teams()).to_csv(db_path + "teams.csv", index=False)

def players_table():
    pd.DataFrame(players.get_players()).to_csv(db_path + "players.csv", index=False)

def games_table(from_year=1990):
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(from_year, 2023)]
    teams = pd.read_csv(db_path + "teams.csv").id.to_list()
    types = ["Regular Season","Pre Season","Playoffs","Preseason"]
    list = []
    for season in tqdm(seasons):
        for team in teams:
            for type in types:
                time.sleep(1)
                for game in teamgamelog.TeamGameLog(season=season,season_type_all_star=type,team_id=team).get_dict()["resultSets"][0]["rowSet"]:
                    list.append({"game_id":game[1],"season":season,"game_type":type,"team_id":game[0],"game_date":game[2],"matchup":game[3],"WL":game[4]})
    games = pd.DataFrame(list).drop_duplicates(ignore_index=True)
    temp = []
    for game in games["game_id"].unique():
        subset = games.loc[games["game_id"]==game,].reset_index()
        if (len(subset.index) != 2):
            continue
        if ("vs." in subset.at[0,"matchup"]):
            temp.append({"game_id":subset.at[0,"game_id"],"season":subset.at[0,"season"],"game_type":subset.at[0,"game_type"],"game_date":subset.at[0,"game_date"],
                        "h_team_id":subset.at[0,"team_id"],"a_team_id":subset.at[1,"team_id"],"matchup":subset.at[0,"matchup"]})
        else:
            temp.append({"game_id":subset.at[0,"game_id"],"season":subset.at[0,"season"],"game_type":subset.at[0,"game_type"],"game_date":subset.at[0,"game_date"],
                        "h_team_id":subset.at[1,"team_id"],"a_team_id":subset.at[0,"team_id"],"matchup":subset.at[1,"matchup"]})
    final = pd.DataFrame(temp)
    final["game_date"] = pd.to_datetime(final["game_date"])
    final = final.sort_values(by="game_date")
    final.to_csv(db_path + "games.csv", index = False)
    

#traditional, advanced, misc, scoring, usage, four_factors
def boxscore_table(type):
    games = pd.read_csv(db_path + "games.csv")
    #valid box score years
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(1996, 2023)]
    string_s = seasons[0]
    for season in seasons:
        string_s += "|" + season
    game_ids = list(games[games["season"].str.contains(string_s[8:])]["game_id"])

    if (exists(db_path+type+"_boxscores_players.csv")):
        team_df = pd.read_csv(db_path+type+"_boxscores_teams.csv")
        player_df = pd.read_csv(db_path+type+"_boxscores_players.csv")
        stored_games = team_df["game_id"].unique()
        for game in stored_games:
            game_ids.remove(game)

    player_table = []
    team_table = []
    for game in tqdm(game_ids):
        #time.sleep(0.15)
        try:
            if (type == "traditional"):
                a = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreTraditional"
            elif (type == "advanced"):
                a = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreAdvanced"
            elif (type == "misc"):
                a = boxscoremiscv3.BoxScoreMiscV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreMisc"
            elif (type == "scoring"):
                a = boxscorescoringv3.BoxScoreScoringV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreScoring"
            elif (type == "usage"):
                a = boxscoreusagev3.BoxScoreUsageV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreUsage"
            elif (type == "four_factors"):
                a = boxscorefourfactorsv3.BoxScoreFourFactorsV3(game_id="00"+str(game)).get_dict()
                bs_containter = "boxScoreFourFactors"
        except AttributeError:
            #print ("Error: 00" + str(game))
            continue
        except (requests.exceptions.ReadTimeout, KeyboardInterrupt):
            if (len(team_table) == 0):
                return (-1)
            df = pd.DataFrame(team_table)
            if (exists(db_path+type+"_boxscores_teams.csv")):
                df = pd.concat([team_df,df])
            df.to_csv(db_path+type+"_boxscores_teams.csv",index=False)

            df = pd.DataFrame(player_table)
            if (exists(db_path+type+"_boxscores_players.csv")):
                df = pd.concat([player_df,df])
            df.to_csv(db_path+type+"_boxscores_players.csv",index=False)
            return (-1)
        for side in ["homeTeam","awayTeam"]:
            temp_t = {"game_id":a[bs_containter]['gameId'],"team_id":a[bs_containter][side]["teamId"]}
            for key in a[bs_containter][side]['statistics']:
                temp_t[key] = a[bs_containter][side]['statistics'][key]
            team_table.append(temp_t)
            for player in a[bs_containter][side]["players"]:
                temp_p = {"game_id":a[bs_containter]['gameId'],"player_id":player['personId'],"team_id":a[bs_containter][side]["teamId"]}
                for key in player['statistics']:
                    temp_p[key] = player['statistics'][key]
                player_table.append(temp_p)

    
    df = pd.DataFrame(team_table)
    if (exists(db_path+type+"_boxscores_teams.csv")):
                df = pd.concat([team_df,df])
    df.to_csv(db_path+type+"_boxscores_teams.csv",index=False)

    df = pd.DataFrame(player_table)
    if (exists(db_path+type+"_boxscores_players.csv")):
        df = pd.concat([player_df,df])
    df.to_csv(db_path+type+"_boxscores_players.csv",index=False)


#Must have scraped nowgoal odds and have nowgoal_odds.csv in intermediates
def odds_table():
    odds = pd.read_csv("./intermediates/nowgoal_odds.csv")
    teams = pd.read_csv(db_path+"teams.csv")
    games = pd.read_csv(db_path+"games.csv")
    id_map = {}
    for i in range(len(teams.index)):
        id_map[teams["full_name"].to_list()[i]] = teams["id"].to_list()[i]
    odds["h_team"] = odds["h_team"].transform(lambda x: id_map[x])
    odds["a_team"] = odds["a_team"].transform(lambda x: id_map[x])
    odds = odds.rename(columns={"h_team":"h_team_id","a_team":"a_team_id"})
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    odds["date"] = pd.to_datetime(odds["date"]).dt.date
    odds = odds.merge(games, how='inner', left_on=["h_team_id","a_team_id","date"], right_on=["h_team_id","a_team_id","game_date"])
    odds = odds.drop(["nowgoal_id","season","game_type","game_date","matchup"],axis=1)
    odds.insert(0, "game_id", odds.pop("game_id"))
    odds = odds.sort_values("date")
    odds = odds.replace(["-"," - "], np.nan)
    odds.to_csv(db_path+"odds.csv",index=False)


#for uploading to git
def compress_db():
    for file in tqdm(os.listdir("./database")):
        a = pd.read_csv(db_path+file)
        a.to_csv("./compressed_database/"+file+".gz",index=False,compression="gzip")

def extract_db():
    for file in tqdm(os.listdir("./compressed_database")):
        a = pd.read_csv("./compressed_database/"+file, compression="gzip")
        a.to_csv("./database/"+file.split(".gz")[0],index=False)

compress_db()
