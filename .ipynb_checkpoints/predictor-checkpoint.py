import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from imblearn.over_sampling import SMOTE

matches = pd.read_csv("matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes

matches["match_importance"] = matches["round"].map({
    "Group stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Final": 5
})

def get_team_experience(df):
    round_counts = {}
    for round_name in df["round"].unique():
        round_counts[f"{round_name}_count"] = df[df["round"] == round_name].groupby("team").size()
    return pd.DataFrame(round_counts).reset_index().fillna(0)

team_experience = get_team_experience(matches)
matches = matches.merge(team_experience, on="team", how="left").fillna(0)


cols = ["gf", "ga", "xg", "xga", "poss", "sh", "sot", "dist", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches[['gf', 'ga']] = matches[['gf', 'ga']].replace('\s*\([^)]*\)', '', regex=True)
matches[cols] = matches[cols].apply(pd.to_numeric, errors='coerce')

def rolling_averages(group, cols, new_cols):
    rolling_stats = group[cols].rolling(3, min_periods=1).mean()
    group[new_cols] = rolling_stats
    return group

matches = matches.groupby("team", as_index=False, group_keys=False).apply(lambda x: rolling_averages(x, cols, new_cols)).reset_index(drop=True)
matches = matches.dropna(subset=new_cols)

matches["target"] = (matches["result"] == "W").astype("int")

def add_head_to_head(df):
    df['h2h_wins'] = 0
    df['h2h_losses'] = 0
    df['h2h_draws'] = 0
    for index, row in df.iterrows():
        team = row['team']
        opponent = row['opponent']
        past_matches = df[(df['team'] == team) & (df['opponent'] == opponent) & (df['date'] < row['date'])]
        df.at[index, 'h2h_wins'] = sum(past_matches['result'] == 'W')
        df.at[index, 'h2h_losses'] = sum(past_matches['result'] == 'L')
        df.at[index, 'h2h_draws'] = sum(past_matches['result'] == 'D')
    return df

matches = add_head_to_head(matches)

def calculate_streaks(df):
    df = df.sort_values(by=['team', 'date'])
    df['win_streak'] = df.groupby('team')['target'].apply(lambda x: x.rolling(window=3, min_periods=1).sum()).reset_index(drop=True)
    df['loss_streak'] = df.groupby('team')['target'].apply(lambda x: (1 - x).rolling(window=3, min_periods=1).sum()).reset_index(drop=True)
    return df

matches = calculate_streaks(matches)

train = matches[matches["date"] < '2021-01-01']
test = matches[matches["date"] > '2021-01-01']

print(f"Number of rows in training set: {len(train)}")
print(f"Number of rows in test set: {len(test)}")

round_columns = [col for col in matches.columns if "_count" in col]
predictors = ["venue_code", "opp_code", "match_importance"] + new_cols + round_columns + ['h2h_wins', 'h2h_losses', 'h2h_draws', 'win_streak', 'loss_streak']

smote = SMOTE(random_state=1)
X_res, y_res = smote.fit_resample(train[predictors], train["target"])

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}

xgb = XGBClassifier(random_state=1)
rs = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=5, random_state=1, n_jobs=-1)
rs.fit(X_res, y_res)

best_model = rs.best_estimator_

cv_scores = cross_val_score(best_model, X_res, y_res, cv=5, scoring='accuracy')
print(f'Cross-validated accuracy scores: {cv_scores}')
print(f'Mean accuracy: {cv_scores.mean()}')

preds = best_model.predict(test[predictors])
pred_probs = best_model.predict_proba(test[predictors])

acc = accuracy_score(test["target"], preds)
precision = precision_score(test["target"], preds)
cm = confusion_matrix(test["target"], preds)
combined = pd.DataFrame(dict(actual_result=test["target"], predicted_result=preds, win_prob=pred_probs[:, 1]), index=test.index)
matches = matches.sort_values("date")
matches.to_csv("matches_rolling.csv", index=False)
combined = combined.merge(matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

class MissingDict(dict):
    __missing__ = lambda self, key: key

country_values = {
    "Belgium": "be Belgium",
    "Sweden": "se Sweden",
    "Netherlands": "nl Netherlands",
    "France": "fr France",
    "Denmark": "dk Denmark",
    "Italy": "it Italy",
    "Czechia": "cz Czechia",
    "Turkiye": "tr Türkiye",
    "Portugal": "pt Portugal",
    "Germany": "de Germany",
    "Romania": "ro Romania",
    "England": "eng England",
    "Norway": "no Norway",
    "Slovenia": "si Slovenia",
    "FR-Yugoslavia": "rs Yugoslavia",
    "Spain": "es Spain",
    "Greece": "gr Greece",
    "Russia": "ru Russia",
    "Switzerland": "ch Switzerland",
    "Croatia": "hr Croatia",
    "Bulgaria": "bg Bulgaria",
    "Latvia": "lv Latvia",
    "Poland": "pl Poland",
    "Austria": "at Austria",
    "Republic-of-Ireland": "ie Rep. of Ireland",
    "Ukraine": "ua Ukraine",
    "Wales": "wls Wales",
    "Albania": "al Albania",
    "Slovakia": "sk Slovakia",
    "Northern-Ireland": "nir Northern Ireland",
    "Iceland": "is Iceland",
    "Hungary": "hu Hungary",
    "Finland": "fi Finland",
    "North-Macedonia": "mk N. Macedonia",
    "Scotland": "sct Scotland"
}

mapping = MissingDict(**country_values)
combined["team_code"] = combined["team"].map(mapping)
merged = combined.merge(combined, left_on=["date", "team_code"], right_on=["date", "opponent"])
merged.to_csv("predictions_euro_2021.csv", index=False)
print(f"Accuracy: {acc}")
print(f"Precision: {precision}")
print("Confusion Matrix:")
print(cm)
