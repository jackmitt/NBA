from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
import time
from datetime import datetime, timedelta

#saves all the game ids to a csv - need to do this to get the url for odds of each game
def get_nowgoal_gameids():
    seasons = [str(yr) + "-" + str(yr+1) for yr in range(2004, 2023)]
    ids = []
    browser = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    browser.maximize_window()
    for season in seasons:
        time.sleep(1)
        #regular season
        browser.get("https://basketball.nowgoal9.com/normal/"+season+"/1")
        time.sleep(1)
        for i in range(10):
            try:
                browser.find_element("xpath", "//*[@id='yearmonthTable2']/tbody/tr/td["+str(i)+"]").click()
            except:
                continue
            time.sleep(1)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            for tr in soup.find(class_="tdlink").find_all("tr"):
                try:
                    ids.append(tr.find_all("td")[3].find("a")["onclick"].split("playertech(")[1].split(",")[0])
                except:
                    pass
        #playoffs
        browser.get("https://basketball.nowgoal9.com/playoffs/"+season+"/1")
        time.sleep(1)
        for i in range(12):
            try:
                browser.find_element("xpath", "//*[@id='playoffsDiv']/table/tbody/tr/td["+str(i)+"]").click()
            except:
                continue
            time.sleep(1)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            for tr in soup.find(class_="tdlink").find_all("tr"):
                try:
                    ids.append(tr.find_all("td")[3].find("a")["onclick"].split("playertech(")[1].split(",")[0])
                except:
                    pass
        #Have to do this for when the play in exists
        try:
            browser.find_element("xpath", "//*[@id='playoffsDiv']/table/tbody/tr[2]/td[1]").click()
        except:
            continue
        time.sleep(1)
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        for tr in soup.find(class_="tdlink").find_all("tr"):
            try:
                ids.append(tr.find_all("td")[3].find("a")["onclick"].split("playertech(")[1].split(",")[0])
            except:
                pass
    df = pd.DataFrame()
    df["id"] = ids
    df = df.drop_duplicates()
    df.to_csv("./intermediates/nowgoal_game_ids.csv",index=False)

#fyi this scrapes hongkong odds not decimal
#you should put this is a loop so that disconnections restart it - it's wrapped in try statements so a failure saves the current data
def get_nowgoal_odds():
    data = []
    ids = pd.read_csv("./intermediates/nowgoal_game_ids.csv")["id"].to_list()
    try:
        old_df = pd.read_csv("./intermediates/nowgoal_odds.csv").drop_duplicates()
        for id in old_df["nowgoal_id"].to_list():
            ids.remove(id)
    except:
        pass
    browser = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    browser.maximize_window()
    for id in ids:
        try:
            browser.get("https://www.nowgoal9.com/oddscompbasket/" + str(id))
            time.sleep(0.75)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
        except:
            try:
                old_df = pd.read_csv("./intermediates/nowgoal_odds.csv")
                df = pd.DataFrame(data)
                df.to_csv("./intermediates/nowgoal_odds.csv",index=False)
                final = pd.concat([old_df, df])
                final.to_csv("./intermediates/nowgoal_odds.csv",index=False)
                return (-1)
            except:
                df = pd.DataFrame(data)
                df.to_csv("./intermediates/nowgoal_odds.csv",index=False)
                return (-1)
        #retrieves as a different time zone
        try:
            date = datetime.strptime(soup.find(class_="time")["data-t"], "%m/%d/%Y %I:%M:%S %p") - timedelta(hours=6)
            temp_dict = {"nowgoal_id":id,"h_team":soup.find_all(class_="sclassName")[0].text[1:],"a_team":soup.find_all(class_="sclassName")[1].text[1:],
                        "date":date,"h_score":soup.find_all(class_="score")[0].text,"a_score":soup.find_all(class_="score")[1].text}
            for row in soup.find(class_="company-comp").find_all("tr"):
                if ('tb-bgcolor' in row["class"] or 'tb-bgcolor1' in row["class"]):
                    pf = row.find("td").text.lower()
                    cols = row.find_all("td")

                    temp_dict[pf+"_open_spread"] = cols[3].find_all("span")[0].text
                    temp_dict[pf+"_h_open_spread_odds"] = cols[2].find_all("span")[0].text
                    temp_dict[pf+"_a_open_spread_odds"] = cols[4].find_all("span")[0].text
                    temp_dict[pf+"_close_spread"] = cols[3].find_all("span")[1].text
                    temp_dict[pf+"_h_close_spread_odds"] = cols[2].find_all("span")[1].text
                    temp_dict[pf+"_a_close_spread_odds"] = cols[4].find_all("span")[1].text

                    temp_dict[pf+"_h_open_ml_odds"] = cols[5].find_all("span")[0].text
                    temp_dict[pf+"_a_open_ml_odds"] = cols[7].find_all("span")[0].text
                    temp_dict[pf+"_h_close_ml_odds"] = cols[5].find_all("span")[1].text
                    temp_dict[pf+"_a_close_ml_odds"] = cols[7].find_all("span")[1].text

                    temp_dict[pf+"_open_total"] = cols[9].find_all("span")[0].text
                    temp_dict[pf+"_h_open_total_odds"] = cols[8].find_all("span")[0].text
                    temp_dict[pf+"_a_open_total_odds"] = cols[10].find_all("span")[0].text
                    temp_dict[pf+"_close_total"] = cols[9].find_all("span")[1].text
                    temp_dict[pf+"_h_close_total_odds"] = cols[8].find_all("span")[1].text
                    temp_dict[pf+"_a_close_total_odds"] = cols[10].find_all("span")[1].text
            data.append(temp_dict)
        except:
            continue
    try:
        old_df = pd.read_csv("./intermediates/nowgoal_odds.csv")
        df = pd.DataFrame(data)
        df.to_csv("./intermediates/nowgoal_odds.csv",index=False)
        final = pd.concat([old_df, df])
        final.to_csv("./intermediates/nowgoal_odds.csv",index=False)
    except:
        df = pd.DataFrame(data)
        df.to_csv("./intermediates/nowgoal_odds.csv",index=False)

while(1): get_nowgoal_odds()