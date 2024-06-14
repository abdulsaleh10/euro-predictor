import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

years = [2021]+ list(range(2016, 1999, -4))
print(years)
standings_url = "https://fbref.com/en/comps/676/2021/2021-European-Championship-Stats"
all_matches = []

import time
for year in years:
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features="html.parser")
    selector = f'table#results{year}6760_overall'
    standings_table = soup.select(selector)[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    if (year != 2000) :
        previous_season = soup.select("a.prev")[0].get("href")
        standings_url = f"https://fbref.com{previous_season}"

    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Men-Stats", "")
        
        data = requests.get(team_url)
        matches = pd.read_html(StringIO(data.text), match="Scores & Fixtures")[0]

        soup = BeautifulSoup(data.text, features="html.parser")
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and '/all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        
        try:
            shooting = pd.read_html(StringIO(data.text), match="Shooting")[0]
            shooting.columns = shooting.columns.droplevel()
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
            
        team_data = team_data[team_data["Comp"] == "UEFA Euro"]
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(10)
        print(team_name)
        print(year)
        print("Waiting")

match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("matches.csv")
