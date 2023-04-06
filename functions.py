"""
Victor Wei
CSE 163 AB
A file that implements the 7 functions for the final project.
These functions involves using pandas, seaborn, and scikit-learn
to conduct operations on basketball for the purpose of predicting the
hof status of players based on their rookie year performance.
"""

import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sns.set()


def process_hof(data):
    '''
    This function accepts the hof dataset and returns the column 'Name.
    It is assumed that the dataframe passed in is not empty and
    contains the required columns.
    '''
    data = data[['Name']]
    return data


def process_draftclass(data):
    '''
    This function accepts the draftclass datasets and returns the columns
    'Player' and 'Pk'. Only the rows with 'Pk' number smaller than or equals
    to 57 will be included.
    It is assumed that the dataframe passed in is not empty and
    contains the required columns.
    '''
    data = data[['Player', 'Pk']]
    result = data[data['Pk'] <= 57]
    return result


def label_draftclass(data):
    '''
    This function accepts merged result of the draftclass datasets and the
    hof_data and labels the hof status of each player in the draftclass
    dataset. The 'Name' column in the returned dataframe is 'hof_status',
    and players inducted into the hof will have the value 'HOF' in the
    hof_status column and the rest of the players will have the value 'NHOF'
    It is assumed that the dataframe passed in is not empty and
    contains the required columns.
    '''
    data.rename(columns={'Name': 'hof_status'}, inplace=True)
    result = data.copy()
    result.loc[(result.hof_status == 0), 'hof_status'] = 'NHOF'
    result.loc[((result.hof_status != 0) &
               ((result.hof_status != 'NHOF'))), 'hof_status'] = 'HOF'
    return result


def basic_stats(player_data, draft_data, year):
    '''
    This function accepts the season_stats dataset, the players drafted of
    a specific year, the year as an int and combines into the basic stats of
    players drafted that year by merging by player. The returned dataframe
    only includes data from the specific year.
    The returned dataframe will include the basic stats columns which includes
    'PTS', 'BLK', 'STL', 'AST', 'G', 'TRB', 'FG%', 'Player', and '3P%'.
    The columns ''PTS', 'BLK', 'STL', 'AST' is divided by the column 'G' to
    find each stats per game.
    The player names are stripped of punctuations and repeated player names of
    the same season have the stats summed or averaged based on the properties
    of the stat. The nan values in 3p% is filled with 0 and the 'G', 'Pk',
    and 'Year' columns are dropped in the end.
    It is assumed that the dataframe passed in is not empty and
    contains the required columns.
    '''
    result = player_data[['PTS', 'BLK', 'STL', 'AST', 'TRB',
                          'FG%', '3P%', 'Player', 'G', 'Year']].copy()
    result['Player'] = result['Player'].str.replace(r'[^\w\s]+', '')
    result = result[result['Year'] == (year + 1)]

    aggregation_functions = {'PTS': 'sum', 'BLK': 'sum', 'STL': 'sum',
                             'AST': 'sum', 'TRB': 'sum', 'FG%': 'mean',
                             '3P%': 'mean', 'G': 'sum', 'Year': 'mean'}
    result = result.groupby(result['Player']).aggregate(aggregation_functions)
    result = pd.merge(result, draft_data, left_on='Player',
                      right_on='Player', how='right')
    result = result.dropna(subset=['PTS'])

    result['PTS'] = result['PTS'] / result['G']
    result['BLK'] = result['BLK'] / result['G']
    result['STL'] = result['STL'] / result['G']
    result['AST'] = result['AST'] / result['G']
    result['TRB'] = result['TRB'] / result['G']
    result = result.drop(['G', 'Pk', 'Year'], axis=1)
    result['3P%'] = result['3P%'].fillna(0)

    return result


def advanced_stats(player_data, draft_data, year):
    '''
    This function accepts the season_stats dataset, the players drafted of
    a specific year, the year as an int and combines into the basic and
    advanced stats of players drafted that year by merging by player. The
    returned dataframe only includes data from the specific year. The returned
    dataframe will include the columns 'PTS', 'BLK', 'STL', 'AST', 'G', 'TRB',
    'PER', 'TS%', 'WS', 'WS/48', 'BPM', 'VORP', 'FG%', 'Player', and '3P%'.
    The columns ''PTS', 'BLK', 'STL', 'AST' is divided by the column 'G' to
    find each stats per game. The player names are stripped of punctuations
    and repeated player names within the same season have the stats summed
    or averaged based on the properties of the stat.
    The nan values in 3p% is filled with 0 and the 'G', 'Pk',
    and 'Year' columns are dropped in the end.
    It is assumed that the dataframe passed in is not empty and
    contains the required columns.
    '''
    result = player_data[['PTS', 'BLK', 'STL', 'AST', 'TRB', 'FG%', '3P%',
                          'PER', 'TS%', 'WS', 'WS/48', 'BPM', 'VORP',
                          'Player', 'G', 'Year']].copy()
    result['Player'] = result['Player'].str.replace(r'[^\w\s]+', '')
    result = result[result['Year'] == (year + 1)]

    aggregation_functions = {'PTS': 'sum', 'BLK': 'sum', 'STL': 'sum',
                             'AST': 'sum', 'TRB': 'sum', 'FG%': 'mean',
                             '3P%': 'mean', 'G': 'sum', 'Year': 'mean',
                             'PER': 'mean', 'TS%': 'mean', 'WS': 'mean',
                             'WS/48': 'mean', 'BPM': 'mean', 'VORP': 'mean'}
    result = result.groupby(result['Player']).aggregate(aggregation_functions)
    result = pd.merge(result, draft_data, left_on='Player',
                      right_on='Player', how='right')
    result = result.dropna(subset=['PTS'])

    result['PTS'] = result['PTS'] / result['G']
    result['BLK'] = result['BLK'] / result['G']
    result['STL'] = result['STL'] / result['G']
    result['AST'] = result['AST'] / result['G']
    result['TRB'] = result['TRB'] / result['G']
    result = result.drop(['G', 'Pk', 'Year'], axis=1)
    result['3P%'] = result['3P%'].fillna(0)

    return result


def predict_hof(data, target_features, target_labels):
    '''
    This function accepts the 1984 draft class rookie season data and
    target_features and target_labels and constructs a DecisionTreeClassifier
    model with max_depth 4 to predict the hof_status column. The 1984 draft
    class data is separated into features and labels for the model. The
    returned results include the accuracy score of the training set,
    the accuracy score according to the target_labels and target_features
    based on the model from the 1984 data, and the final prediction series.
    '''
    data = data.drop('Player', axis=1)
    data.fillna(0)
    features = data.loc[:, data.columns != 'hof_status']
    labels = data['hof_status']

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(features, labels)

    train_pred = model.predict(features)
    train_acc_score = accuracy_score(labels, train_pred)
    tar_pred = model.predict(target_features)
    tar_acc_score = accuracy_score(target_labels, tar_pred)

    return train_acc_score, tar_acc_score, tar_pred


def predict_hof_2016(data, target_features):
    '''
    This function accepts the 1984 draft class rookie season data and
    target_features and constructs a DecisionTreeClassifier model with
    max_depth 4 to predict the hof_status column of the target_features.
    The 1984 draft class data is separated into features and labels for
    the model. The returned results include the accuracy score of the
    training set and the final prediction series for the target_features.
    This function is only inteded for 2016 prediction.
    '''
    data = data.drop('Player', axis=1)
    data.fillna(0)
    features = data.loc[:, data.columns != 'hof_status']
    labels = data['hof_status']

    model = DecisionTreeClassifier(max_depth=4)
    model.fit(features, labels)

    train_pred = model.predict(features)
    train_acc_score = accuracy_score(labels, train_pred)
    tar_pred = model.predict(target_features)

    return train_acc_score, tar_pred
