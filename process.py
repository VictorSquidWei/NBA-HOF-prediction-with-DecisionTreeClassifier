"""

A file that implements data procesing and modeling construction process of the
final project. The purpose of this project is to eventually predict the future
hof status of the players in the draft class 2016 from stats from the draft
class of 1996 and 1984.
The data used in this file came from three different sources. The first one
is the season_stats.csv is from the kaggle dataset listed in the report that
records all player data since 1950, I have three draft class name lists from
basketballreference.com for the years 1996, 1984, and 2016. I also have a HOF
name list from hispanosnba.com.
"""

import pandas as pd
import functions


def main():
    '''
    Main method. Loads the dataset provided and calls all the functions
    defined in functions.py in the logical process presented in the report
    '''
    hof_data = pd.read_csv('../Final_project/hof.csv')
    draft_1996 = pd.read_csv('../Final_project/1996DraftClass.csv')
    draft_1984 = pd.read_csv('../Final_project/1984DraftClass.csv')
    draft_2016 = pd.read_csv('../Final_project/2016DraftClass.csv')
    player_stats = pd.read_csv('../Final_project/Seasons_Stats.csv')

    draft_2016 = functions.process_draftclass(draft_2016)
    draft_1984 = functions.process_draftclass(draft_1984)
    draft_1996 = functions.process_draftclass(draft_1996)
    hof_data = functions.process_hof(hof_data)

    hof_1996 = pd.merge(hof_data, draft_1996, left_on='Name',
                        right_on='Player', how='right').fillna(0)
    hof_1984 = pd.merge(hof_data, draft_1984, left_on='Name',
                        right_on='Player', how='right').fillna(0)

    hof_1996 = functions.label_draftclass(hof_1996)
    hof_1984 = functions.label_draftclass(hof_1984)

    ba_stats_1996 = functions.basic_stats(player_stats, hof_1996, 1996)
    ba_stats_1984 = functions.basic_stats(player_stats, hof_1984, 1984)

    adv_stats_1996 = functions.advanced_stats(player_stats, hof_1996, 1996)
    adv_stats_1984 = functions.advanced_stats(player_stats, hof_1984, 1984)

    print(adv_stats_1984)
    print("Draft class 1996 Basic Stats: ")
    print(ba_stats_1996)
    ba_ft_1996 = ba_stats_1996.loc[:, ba_stats_1996.columns != 'hof_status']
    ba_lb_1996 = ba_stats_1996['hof_status']
    ba_ft_1996 = ba_ft_1996.drop('Player', axis=1)
    print("Basic Stats Prediction Accuracy: ")
    print(functions.predict_hof(ba_stats_1984, ba_ft_1996, ba_lb_1996))

    print("Draft class 1996 Advanced Stats: ")
    print(adv_stats_1996)
    adv_ft_1996 = adv_stats_1996.loc[:, adv_stats_1996.columns != 'hof_status']
    adv_lb_1996 = adv_stats_1996['hof_status']
    adv_ft_1996 = adv_ft_1996.drop('Player', axis=1)
    print("Advanced Stats Prediction Accuracy: ")
    print(functions.predict_hof(adv_stats_1984, adv_ft_1996, adv_lb_1996))
    adv_stats_2016 = functions.advanced_stats(player_stats, draft_2016, 2016)
    print("Draft class 2016 Advanced Stats: ")
    print(adv_stats_2016)
    adv_stats_2016 = adv_stats_2016.drop('Player', axis=1)
    print("Draft Class 2016 Advanced Stats Prediction: ")
    print(functions.predict_hof_2016(adv_stats_1984, adv_stats_2016))


if __name__ == '__main__':
    main()
