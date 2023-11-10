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
from tqdm import tqdm
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

    player_table = []
    team_table = []
    game_ids = games[games["season"].str.contains(string_s[8:])]["game_id"]
    for game in tqdm(game_ids):
        #time.sleep(0.5)
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
            print ("Error: 00" + str(game))
            continue
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
    df.to_csv(db_path+type+"_boxscores_teams.csv",index=False)

    df = pd.DataFrame(player_table)
    df.to_csv(db_path+type+"_boxscores_players.csv",index=False)

boxscore_table("advanced")
