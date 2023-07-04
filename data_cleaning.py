#%%
# Import libraries
import pandas as pd
import numpy as np
import re
import nfl_data_py as nfl
import os
import pathlib
os.chdir(pathlib.Path(__file__).parent.absolute())

pd.set_option('display.max_columns', None)
years = [2020]
def get_play_by_play_data(years):
    _pbp : pd.DataFrame = nfl.import_pbp_data(years)
    pbp = _pbp.copy()
    pbp_cols = []
    with open('pandas_columns/pbp_data_columns.txt', 'r') as f:
        for line in f:
            pbp_cols.append(line.strip())

    pbp = pbp[pbp_cols] # Keep only potentially relevant columns

    # Add odds.
    pbp = pbp.assign(
        vegas_odds = lambda df: df.apply(lambda row: 1/row.vegas_wp if row.vegas_wp != 0 else 0, axis='columns'),
        odds = lambda df: df.apply(lambda row: 1/row.wp if row.wp != 0 else 0, axis='columns'),
        ) # odds built-in
    pbp = pbp[~pbp.odds.isna() & ~pbp.posteam_type.isnull()] # gets rid of END OF PLAY plays.
    pbp["score_differential"] = pbp["home_score"] - pbp["away_score"]
    pbp = pbp.drop(['home_score', "away_score"], axis='columns')
    pbp.insert(5, "end_score_differential", pbp.pop("score_differential"))
    return pbp

def aggregate_play_to_game(play_by_play):
    agg_pbp = play_by_play.copy()
    agg_pbp = agg_pbp.fillna(0)

    agg_pbp = agg_pbp[[_col for _col in play_by_play.columns if "total" not in _col]] # Or just straight up take them out
    agg_foos = dict.fromkeys(agg_pbp, 'sum')
    # agg_foos.update(dict.fromkeys([col for col in pbp.columns if "total" in col], 'last')) # Stats where only the last value really matters.
    agg_foos.update(dict.fromkeys(agg_pbp.columns[agg_pbp.dtypes.eq(object)], 'first')) # Basically everything other than floats, which is not very summable.
    agg_foos.update(dict.fromkeys(['season', 'week', 'vegas_odds', 'odds', 'end_score_differential'], 'first'))

    # Aggregate per game (game-by-game)
    gbg = agg_pbp.groupby(['season', 'game_id', 'posteam_type'], as_index= False).agg(agg_foos)
    gbg.insert(0, 'season', gbg.pop('season')) # Move season to front for aesthetic reasons :)
    gbg.loc[gbg["posteam_type"] == "away", "end_score_differential"] = -1 * gbg.loc[gbg["posteam_type"] == "away", "end_score_differential"]
    return gbg

def get_player_stats_by_week(years):
    players_gbg = nfl.import_weekly_data(years)
    player_cols = []
    with open('pandas_columns/player_data_columns.txt', 'r') as f:
        for line in f:
            player_cols.append(line.strip())

    players_gbg = players_gbg[player_cols]
    return players_gbg

def get_team_info_for_week(_df : pd.DataFrame):
    df = _df[["game_id", "season", "week", "home_team", "away_team", "posteam_type"]].copy()
    df["team"] = ""
    df["op_team"] = ""
    df.loc[df["posteam_type"] == 'home', "team"] = df["home_team"]
    df.loc[df["posteam_type"] == 'home', "op_team"] = df["away_team"]
    df.loc[df["posteam_type"] == 'away', "team"] = df["away_team"]
    df.loc[df["posteam_type"] == 'away', "op_team"] = df["home_team"]
    df["home_team"] = 0
    df.loc[df["posteam_type"] == 'home', "home_team"] = 1
    return df[["season", "week", "game_id", "team", "op_team", "home_team"]]

def aggregate_team_data(players_by_game, game_by_game):
    agg_players = players_by_game.copy()
    agg_players = agg_players.fillna(0)

    agg_foos = dict.fromkeys(agg_players, 'sum')
    agg_foos.update(dict.fromkeys(agg_players.columns[agg_players.dtypes.eq(object)], 'first'))
    agg_foos.update({'season' : 'first', 'week' : 'first'})
    team_gbg = agg_players.groupby(['season', 'week', 'recent_team'], as_index= False).agg(agg_foos)
    team_gbg.rename(columns= {'recent_team': 'team'}, inplace=True)
    team_info = get_team_info_for_week(game_by_game)
    team_gbg = pd.merge(team_info, team_gbg, on=["season", "week", "team"])
    team_gbg = team_gbg.drop([c for c in team_gbg.columns if "share"in c], axis='columns') # drop shares, because they're always 1 when considering whole team
    return team_gbg

def append_opponent_data(df : pd.DataFrame):
    d = df.copy()
    groups = ["season", "week", "game_id"]
    gd = d.groupby(groups, as_index= False)
    no_repeat = ["season", "week", "game_id", "team", "op_team", "home_game", "end_score_differential"]
    use = [col for col in d.columns if col not in no_repeat]
    op_use = ['op_'+col for col in use]
    def mix_rows(df : pd.DataFrame):
        one = df.iloc[0].copy()
        two = df.iloc[1].copy()
        one_to_two = one.rename({key : val for key, val in zip(use, op_use)})
        two_to_one = two.rename({key : val for key, val in zip(use, op_use)})
        out_one = pd.concat([one, two_to_one[op_use]], axis= 0)
        out_two = pd.concat([two, one_to_two[op_use]], axis= 0)
        out = pd.DataFrame([out_one, out_two])
        return out

    out = gd.apply(mix_rows).reset_index(drop= True)
    out = out.rename({'home_team' : 'home_game'}, axis="columns")
    return out

def merge_game_and_team(game, team):
    # Merge into final DS
    game = game.drop(["home_team", "away_team"], axis="columns")
    game["home_team"] = [1 if _a == 'home' else 0 for _a in game.posteam_type]
    game = game.drop(["posteam_type"], axis="columns")
    df = pd.merge(team, game, on= ["season", "week", "game_id", "home_team"])
    df.insert(6, "end_score_differential", df.pop("end_score_differential"))
    # df.insert(7, "vegas_odds", df.pop("vegas_odds"))
    # df.insert(8, "odds", df.pop("odds"))
    return df

def create_exp_weighted_avgs(df, span):
    # Create a copy of the df with only the game id and the team - we will add cols to this df
    ema_features = df[['season', 'week', 'game_id', 'team', 'op_team', 'home_game']].copy()

    feature_cols = []
    with open('pandas_columns/feature_columns.txt', 'r') as f:
        for line in f:
            feature_cols.append(line.strip())
            feature_cols.append('op_' + line.strip())
    feature_names = [col for col in df.columns if col in feature_cols] # Get a list of columns we will iterate over

    _df = df.loc[:, ~df.columns.isin(['game_id', 'op_team'])].copy()
    _df = _df.fillna(0)
    new_cols = []
    for feature_name in feature_names:
        feature_ema = (_df.groupby('team')[feature_name]
                         .transform(lambda row: (row.ewm(span=span)
                                                    .mean()
                                                    .shift(1))))
        new_cols.append(feature_ema)

    out = pd.concat([ema_features, *new_cols], axis=1)
    out = out.rename({old_name : new_name for old_name, new_name in zip(feature_cols, ['f_'+col for col in feature_cols])}, axis='columns')
    return out

def get_form_btwn_teams(df, span, performance_categories = None):
    if performance_categories is None:
        performance_categories = ['end_score_differential', 'passing_epa', 'qb_epa', 'xyac_epa', 'air_epa', 'yac_epa', 'rushing_epa']
    form_btwn_teams = df[['game_id', 'team', 'op_team', *performance_categories]].copy()

    # For wins against eachother
    form_btwn_teams['f_form_past_5_btwn_teams'] = \
    (df.assign(win=lambda df: df.apply(lambda row: 1 if row.end_score_differential > 0 else 0, axis='columns'))
                .groupby(['team', 'op_team'])['win']
                .transform(lambda row: row.rolling(span).mean().shift() * span)
                .fillna(0))

    for cat in performance_categories:
        form_btwn_teams[f'f_form_{cat}_btwn_teams'] = (df.groupby(['team', 'op_team'])[cat]
                                                                .transform(lambda row: row.rolling(span).mean().shift())
                                                                .fillna(0))
    return form_btwn_teams

# Elo Sanity check
# Define a function which finds the elo for each team in each game and returns a dictionary with the game ID as a key and the
# elos as the key's value, in a list. It also outputs the probabilities and a dictionary of the final elos for each team
def elo_applier(df, k_factor, verbose= False):
    # Initialise a dictionary with default elos for each team
    elo_dict = {team: 1500 for team in df['team'].unique()}
    elos, elo_probs = {}, {}

    # Get a home and away dataframe so that we can get the teams on the same row
    home_df = df.loc[df.home_game == 1, ['team', 'game_id', 'end_score_differential', 'home_game']].rename(columns={'team': 'home_team'})
    away_df = df.loc[df.home_game == 0, ['team', 'game_id']].rename(columns={'team': 'away_team'})

    df = (pd.merge(home_df, away_df, on='game_id')
            .sort_values(by='game_id')
            .drop_duplicates(subset='game_id', keep='first')
            .reset_index(drop=True))

    # Loop over the rows in the DataFrame
    for index, row in df.iterrows():
        # Get the Game ID
        game_id = row['game_id']

        # Get the margin
        margin = row['end_score_differential']

        # If the game already has the elos for the home and away team in the elos dictionary, go to the next game
        if game_id in elos.keys():
            continue

        # Get the team and opposition
        home_team = row['home_team']
        away_team = row['away_team']

        # Get the team and opposition elo score
        home_team_elo = elo_dict[home_team]
        away_team_elo = elo_dict[away_team]

        # Calculated the probability of winning for the team and opposition
        prob_win_home = 1 / (1 + 10**((away_team_elo - home_team_elo) / 400))
        prob_win_away = 1 - prob_win_home

        # Add the elos and probabilities our elos dictionary and elo_probs dictionary based on the Game ID
        elos[game_id] = [home_team_elo, away_team_elo]
        elo_probs[game_id] = [prob_win_home, prob_win_away]

        # Calculate the new elos of each team
        if margin > 0: # Home team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away)
        elif margin < 0: # Away team wins; update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away)
        elif margin == 0: # Drawn game' update both teams' elo
            new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away)

        # Update elos in elo dictionary
        elo_dict[home_team] = new_home_team_elo
        elo_dict[away_team] = new_away_team_elo

    if verbose:
        for team in sorted(elo_dict, key=elo_dict.get)[::-1]:
            print(team, elo_dict[team])

    return elos, elo_probs, elo_dict

# In this case only elos
def append_extra_features(features, elos):
    one_line_cols = ['game_id', 'team', 'home_game'] + [col for col in features if col.startswith('f_')]

    # Get all features onto individual rows for each match
    features_one_line = (features.loc[features.home_game == 1, one_line_cols]
                        .rename(columns={'team': 'home_team'})
                        .drop(columns='home_game'))
                        #  .pipe(pd.merge, (features.loc[features.home_game == 0, one_line_cols]
                        #                           .drop(columns='home_game')
                        #                           .rename(columns={'team': 'away_team'})
                        #                           .rename(columns={col: col+'_away' for col in features.columns if col.startswith('f_')})), on='game_id')
                        # )

    # Add extra created features - elo
    features_one_line = (features_one_line.assign(f_elo_home=lambda df: df.game_id.map(elos).apply(lambda x: x[0]),
                                                f_elo_away=lambda df: df.game_id.map(elos).apply(lambda x: x[1]))
                                        #   .pipe(pd.merge, form_btwn_teams.loc[nfl_data.home_game == 1, ['game_id', 'week']], on=['game_id'])
                                        .dropna()
                                        .reset_index(drop=True))

    ordered_cols = [col for col in features_one_line if col[:2] != 'f_'] + [col for col in features_one_line if col.startswith('f_')]

    feature_df = features_one_line[ordered_cols]
    return feature_df

def reduce_feature_df(feature_df : pd.DataFrame, nfl_data) -> pd.DataFrame:
    # Create differential df - this df is the home features - the away features
    o_diff_cols = []
    with open('pandas_columns/feature_columns.txt', 'r') as f:
        for line in f:
            o_diff_cols.append(line.strip())
            # o_diff_cols.append('op_' + line.strip())
    diff_cols = [col for col in o_diff_cols if (("wpa" in col and "wp" not in col) or ("wp" not in col))]
    # diff_cols = [col for col in feature_df.columns if col + '_away' in feature_df.columns and col != 'f_odds' and col.startswith('f_')]
    non_diff_cols = [col for col in feature_df.columns if (col[2:] not in diff_cols and col[5:] not in diff_cols)]
    # non_diff_cols.append('game_id')

    diff_df = feature_df[non_diff_cols].copy()

    for col in diff_cols:
        diff_df['f_' + col + '_diff'] = feature_df['f_' + col] - feature_df['f_op_' + col]

    odds = nfl_data[['game_id', 'team', 'op_team', 'home_game', 'odds', 'vegas_odds']]
    home_odds = (odds[odds.home_game == 1]
                .assign(f_current_odds_prob=lambda df: 1 / df.odds)
                .rename(columns={'team': 'home_team'})
                .rename(columns={'op_team' : 'away_team'})
                .drop(columns=['home_game', 'odds']))

    away_odds = (odds[odds.home_game == 0]
                .assign(f_current_odds_prob_away=lambda df: 1 / df.odds)
                .rename(columns={'team': 'away_team'})
                .drop(columns=['home_game', 'odds']))

    diff_df = (diff_df.pipe(pd.merge, home_odds, on=['game_id', 'home_team'])
                .pipe(pd.merge, away_odds, on=['game_id', 'away_team'])
                .drop(['home_team', 'away_team', 'op_team'], axis="columns")
                .assign(season= diff_df.game_id.apply(lambda row: int(row.split('_')[0]))))
    return diff_df


def get_data(years, performance_comparisson_between_teams, rolling_average_span):
    """
    gets the input data for years, might lose some of the first datapoints because of rolling averages
    """
    # Load Play-By-Play data and filter columns.
    pbp = get_play_by_play_data(years)
    gbg = aggregate_play_to_game(pbp)
    players_gbg = get_player_stats_by_week(years)
    team_gbg = aggregate_team_data(players_gbg, gbg)
    # Check historical performance between teams
    merged_data = merge_game_and_team(gbg, team_gbg)
    nfl_data = append_opponent_data(merged_data)
    # Feature Creation
    ## A rolling average for the features would be a good idea probably
    features = create_exp_weighted_avgs(nfl_data, rolling_average_span)
    # Get matchup specific history and merge to features -
    # They are going to be NaN or 0 for the first datapoints, because of the rolling average
    form_btwn_teams = get_form_btwn_teams(nfl_data, rolling_average_span, performance_comparisson_between_teams)
    # Merge to our features df
    features = pd.merge(features, form_btwn_teams.drop(columns=['end_score_differential']), on=['game_id', 'team', 'op_team'])
    # Use the elo applier function to get the elos and elo probabilities for each game - we will map these later
    elos, probs, elo_dict = elo_applier(nfl_data, 30)
    feature_df = append_extra_features(features, elos)
    data = reduce_feature_df(feature_df, nfl_data)


    match_results = (nfl_data[['season', 'game_id', 'team', 'end_score_differential']]
                     .assign(result=lambda df: df.apply(
                                    lambda row: 1 if row['end_score_differential'] > 0 else 0, axis=1)
                                    )
                    )
    # Labels
    labels = (match_results
              .groupby('game_id', as_index= False).first())
    labels = (labels
              .loc[labels.game_id.isin(data.game_id), ["season", "result"]]
              .reset_index(drop=True)
            )
    
    return {'X' : data, 'Y' : labels, 'info' : nfl_data}

if __name__ == "__main__":
    years = [2020]
    perf_cat = ['end_score_differential', 'passing_epa', 'qb_epa', 'xyac_epa', 'air_epa', 'yac_epa', 'rushing_epa']
    data = get_data(years, perf_cat, 5)