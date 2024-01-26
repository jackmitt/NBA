from nba_api.stats.static import teams
from nba_api.stats.static import players
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.endpoints import boxscoretraditionalv3
from nba_api.stats.endpoints import boxscoreadvancedv3
from nba_api.stats.endpoints import boxscoremiscv3
from nba_api.stats.endpoints import boxscorescoringv3
from nba_api.stats.endpoints import boxscoreusagev3
from nba_api.stats.endpoints import boxscorefourfactorsv3
from nba_api.stats.endpoints import playbyplayv3
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


def play_by_play():
    games = pd.read_csv(db_path + "games.csv")
    #valid years
    seasons = [str(yr) + "-" + str(yr+1)[2:4] for yr in range(1996, 2023)]
    string_s = seasons[0]
    for season in seasons:
        string_s += "|" + season
    game_ids = list(games[games["season"].str.contains(string_s[8:])]["game_id"])

    if (exists(db_path+"play_by_play.csv")):
        df_old = pd.read_csv(db_path+"play_by_play.csv")
        stored_games = df_old["game_id"].unique()
        for game in stored_games:
            game_ids.remove(game)

    table = []
    for game in tqdm(game_ids):
        try:
            a = playbyplayv3.PlayByPlayV3(game_id="00"+str(game)).get_dict()
        except (requests.exceptions.ReadTimeout, KeyboardInterrupt):
            if (len(table) == 0):
                return (-1)
            df = pd.DataFrame(table)
            if (exists(db_path+"play_by_play.csv")):
                df = pd.concat([df_old,df])
            df.to_csv(db_path+"play_by_play.csv",index=False)

            return (-1)
        except:
            #print ("Error: 00" + str(game))
            continue
        for action in a['game']['actions']:
            temp_t = {'game_id':a['game']['gameId']}
            for key in action:
                temp_t[key] = action[key]
            table.append(temp_t)


    
    df = pd.DataFrame(table)
    if (exists(db_path+"play_by_play.csv")):
        df = pd.concat([df_old,df])
    df.to_csv(db_path+"play_by_play.csv",index=False)
    return (1)

#Uses play by play data to get data about every unique lineup during the game
#need to come up with a way to validate
def lineups_on_court():
    pbp = pd.read_csv('C:/Users/jackj/OneDrive/Desktop/play_by_play.csv')
    box_score = pd.read_csv(db_path+'traditional_boxscores_players.csv')
    game_ids = list(pbp["game_id"].unique())

    #players_df = pd.read_csv(db_path+'players.csv')
    #player_map = dict.fromkeys(players_df[''])

    table = []

    for gid in tqdm(game_ids):
        game_pbp = pbp.loc[pbp['game_id']==gid,].reset_index(drop=True)
        game_bs = box_score.loc[box_score['game_id']==gid,]

        sub_events = game_pbp.loc[game_pbp['actionType']=="Substitution", ].copy()

        sub_events['sub_id'] = sub_events.groupby(['clock','period']).ngroup()

        h_team = game_bs['team_id'].unique()[0]
        a_team = game_bs['team_id'].unique()[1]

        h_team_players = []
        for index, row in game_bs.loc[game_bs['team_id'] == h_team].iterrows():
            if (not pd.isnull(row['minutes'])):
                h_team_players.append(row['player_id']) 
        a_team_players = []
        for index, row in game_bs.loc[game_bs['team_id'] == a_team].iterrows():
            if (not pd.isnull(row['minutes'])):
                a_team_players.append(row['player_id'])

        h_pbp = game_pbp.loc[game_pbp['teamId']==h_team,]
        a_pbp = game_pbp.loc[game_pbp['teamId']==a_team,]

        dict = {'game_id':gid,'h_id':h_team,'a_id':a_team,'start':0,'end':-1}

        #starters
        for i in range(1,6):
            h_lineup = list(game_bs.loc[game_bs['team_id']==h_team,]['player_id'])
            dict['h_player_'+str(i)] = h_lineup[i-1]
        for i in range(1,6):
            a_lineup = list(game_bs.loc[game_bs['team_id']==a_team,]['player_id'])
            dict['a_player_'+str(i)] = a_lineup[i-1]
    

        new_lineup = False

        for index in range(len(game_pbp.index)):
            if (game_pbp.at[index, 'actionType'] == 'period' and game_pbp.at[index, 'subType'] == 'start' and game_pbp.at[index, 'period'] != 1):
                dict['end'] = (game_pbp.at[index, 'period']-1) * 720
                table.append(dict)
                dict = {'game_id':gid,'h_id':h_team,'a_id':a_team,'start':(game_pbp.at[index, 'period']-1) * 720,'end':-1}
                h_players_found = []
                a_players_found = []
                #new_lineup is true when 2,3,4,OTs periods begin and we do not know who started the quarter on the court; switched to False when we know
                new_lineup = True
            #bench players can get technical fouls
            if (new_lineup and not (game_pbp.at[index, 'actionType'] == 'Foul' and game_pbp.at[index, 'subType'] == 'Technical')):
                if (game_pbp.at[index,'teamId'] == h_team and game_pbp.at[index,'personId'] not in h_players_found):
                    h_players_found.append(game_pbp.at[index,'personId'])
                    dict['h_player_'+str(len(h_players_found))] = game_pbp.at[index,'personId']
                if (game_pbp.at[index,'teamId'] == a_team and game_pbp.at[index,'personId'] not in a_players_found):
                    a_players_found.append(game_pbp.at[index,'personId'])
                    dict['a_player_'+str(len(a_players_found))] = game_pbp.at[index,'personId']
                if (len(a_players_found) == 5 and len(h_players_found) == 5):
                    new_lineup = False

            if (game_pbp.at[index, 'actionType'] == 'Substitution'):
                if (game_pbp.at[index,'period'] <= 4):
                    sec_elapsed_game = 720*(game_pbp.at[index,'period']) - float(game_pbp.at[index,'clock'].split("PT")[1].split("M")[0]) * 60 - float(game_pbp.at[index,'clock'].split("M")[1].split("S")[0])
                else:
                    sec_elapsed_game = 2880 + 300*(game_pbp.at[index,'period']) - float(game_pbp.at[index,'clock'].split("PT")[1].split("M")[0]) * 60 - float(game_pbp.at[index,'clock'].split("M")[1].split("S")[0])

                #This handles when we reach the first substitution of the quarter and we still do not know who started the quarter
                #We iterate through every remaining play for the rest of the period to find the starters - we can never find them if they play all quarter without any contribution
                if (new_lineup):
                    h_subbed_on = []
                    a_subbed_on = []
                    end_period_index = list(np.where(game_pbp['subType'] == 'end'))[0][game_pbp.at[index,'period']-1]
                    for j in range(index, end_period_index):
                        if (not (game_pbp.at[j, 'actionType'] == 'Foul' and game_pbp.at[j, 'subType'] == 'Technical')):
                            if (game_pbp.at[j,'teamId'] == h_team and game_pbp.at[j,'personId'] not in h_players_found and game_pbp.at[j,'personId'] not in h_subbed_on):
                                h_players_found.append(game_pbp.at[j,'personId'])
                                dict['h_player_'+str(len(h_players_found))] = game_pbp.at[j,'personId']
                            if (game_pbp.at[j,'teamId'] == a_team and game_pbp.at[j,'personId'] not in a_players_found and game_pbp.at[j,'personId'] not in a_subbed_on):
                                a_players_found.append(game_pbp.at[j,'personId'])
                                dict['a_player_'+str(len(a_players_found))] = game_pbp.at[j,'personId']
                            if (len(a_players_found) == 5 and len(h_players_found) == 5):
                                new_lineup = False
                                break
                        
                        if (game_pbp.at[j, 'actionType'] == 'Substitution'):
                            replace_name = game_pbp.at[j,'description'].split("SUB: ")[1].split(" FOR")[0]
                            replace_team = game_pbp.at[j,'teamId']
                            if (' ' in replace_name):
                                if (replace_team == h_team):
                                    replace_id = h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
                                    if (len(h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                                        print ("ERROR: Duplicate Names")
                                else:
                                    replace_id = a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
                                    if (len(a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                                        print ("ERROR: Duplicate Names")
                            else:
                                if (replace_team == h_team):
                                    replace_id = h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()[0]
                                    if (len(h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                                        print ("ERROR: Duplicate Names")
                                else:
                                    replace_id = a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()[0]
                                    if (len(a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                                        print ("ERROR: Duplicate Names")
                            if (replace_team == h_team):
                                h_subbed_on.append(replace_id)
                            else:
                                a_subbed_on.append(replace_id)
                
                #when multiple subs happen consecutively
                if (not (game_pbp.at[index-1, 'actionType'] == 'Substitution' and game_pbp.at[index-1, 'clock'] == game_pbp.at[index, 'clock'])):
                    dict['end'] = sec_elapsed_game
                    table.append(dict.copy())

                    dict['start'] = sec_elapsed_game
                    dict['end'] = -1

                #finds what position in the dict the player being subbed out is
                key = list(dict.keys())[list(dict.values()).index(game_pbp.at[index,'personId'])]
                replace_name = game_pbp.at[index,'description'].split("SUB: ")[1].split(" FOR")[0]
                replace_team = game_pbp.at[index,'teamId']
                if ('.' in replace_name):
                    if (replace_team == h_team):
                        replace_id = h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
                        if (len(h_pbp.loc[h_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                            print ("ERROR: Duplicate Names")
                    else:
                        replace_id = a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()[0]
                        if (len(a_pbp.loc[a_pbp['playerNameI']==replace_name, ]['personId'].unique()) > 1):
                            print ("ERROR: Duplicate Names")
                else:
                    if (replace_team == h_team):
                        print (gid, replace_name)
                        replace_id = h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()[0]
                        if (len(h_pbp.loc[h_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                            print ("ERROR: Duplicate Names")
                    else:
                        replace_id = a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()[0]
                        if (len(a_pbp.loc[a_pbp['playerName']==replace_name, ]['personId'].unique()) > 1):
                            print ("ERROR: Duplicate Names")
                dict[key] = replace_id
            
        end_period = list(game_pbp['period'])[-1]
        dict["end"] = 2880 + (end_period-4)*300
        table.append(dict.copy())

    df = pd.DataFrame(table)
    df.to_csv(db_path+'unique_lineups.csv',index=False)


            

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
