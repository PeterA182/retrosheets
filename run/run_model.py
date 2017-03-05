from __future__ import division

__author__ = 'paltamura'

import sys
import gc
import os

import datetime as dt
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

sys.path.append("E:\\Models\\retrosheets\\data\\")
from prepRetroData import standardPrep
from toolkit import compress_columns

sys.path.append("E:\\Models\\retrosheets\\model\\")
from features import slump_metrics, streak_metrics


#
# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----
# Methods


def fillna_process(df, col_list):
    total_masks = [df[col].isnull() for col in features]
    total_masks = reduce(np.logical_or, total_masks)
    fillna_table = df.loc[total_masks, :]
    col_na_value = {}
    for col in col_list:
        col_na_value[col] = np.mean(fillna_table[col])

    # Fillna values in real table
    for col in col_list:
        print (" "*20) + col
        print (" "*20) + str(len(df.loc[df[col].isnull(), :]))
        df[col].fillna(col_na_value[col], inplace=True)
        print (" "*20) + str(len(df.loc[df[col].isnull(), :]))

    return df


if __name__ == "__main__":

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ----
    # Paths
    projectPath = 'E:\\Models\\retrosheets\\data\\'
    resultsPath = projectPath + 'results\\'

    # Features
    features = [
        'hits', 'strikeouts', 'hits_slump_count', 'hits_slump_days',
        'hits_slump_avg_len', 'strikeouts_slump_count',
        'strikeouts_slump_days', 'strikeouts_slump_avg_len'
    ]

    # Read in with standard preparation
    data = pd.read_csv(
        projectPath + "prepped\\fullBattingData_2010-2016.csv"
    )

    # Groupby up to game level
    data = data.groupby(
        ['mlb_season', 'game_id', 'game_date', 'batter'],
        as_index=False
    ).agg({'hit_flag': {'hits': np.sum,
                        'batting_avg': np.mean},
           'strikeout_flag': {'strikeouts': np.sum}})
    data = compress_columns(data)

    # Establish season level player table
    playerTable = data.groupby(
        ['mlb_season', 'batter'],
        as_index=False
    ).agg({'hits': np.sum,
           'strikeouts': np.sum})

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----
    # Begin adding features
    df_slump_metrics = slump_metrics(
        df=data, metrics_cols=['hits', 'strikeouts'], threshholds=[15, 85]
    )
    pre_len = len(playerTable)
    playerTable = pd.merge(
        left=playerTable,
        right=df_slump_metrics,
        on=['mlb_season', 'batter'],
        how='left'
    )
    if len(playerTable) != pre_len:
        raise Exception("Merge of type \'left\' has added rows.")
    del df_slump_metrics
    gc.collect()

    # Reestablish playerTable columns
    playerTable = playerTable.loc[:, [
        'mlb_season', 'batter', 'hits', 'strikeouts',
        'hits_slump_count', 'hits_slump_days', 'hits_slump_avg_len',
        'strikeouts_slump_count', 'strikeouts_slump_days',
        'strikeouts_slump_avg_len'
    ]]

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Split into dataframe year: data
    print "All Years: " + ", ".join(
        str(x) for x in sorted(list(set(playerTable['mlb_season'])))
    )
    years = list(set(playerTable['mlb_season']))
    yrs_dict = {
        yr: playerTable.loc[playerTable['mlb_season'] == yr, :] for yr in years
    }
    del playerTable
    gc.collect()

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Append following season hits to all years
    # (except last -- it would have no follower)
    playerTable = pd.DataFrame()
    span = sorted(list(yrs_dict.keys()))[:-1]
    print "Years to add fllwng season hits: " + ", ".join(
        str(x) for x in span
    )
    for yr in span:
        base = yrs_dict[yr]
        following = yrs_dict[yr+1]
        following = following.loc[:, ['batter', 'hits']]
        following.rename(columns={'hits': 'nextYearHits'}, inplace=True)
        pre_len = len(base)
        base = pd.merge(
            left=base,
            right=following,
            how='left',
            on=['batter']
        )
        if len(base) != pre_len:
            raise Exception("Merge of type \'left\' has added rows.")
        playerTable = playerTable.append(base)

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Split intro training
    print "Train Years: " + ", ".join(str(x) for x in yrs_dict.keys()[:-1])
    print "Test Years:  " + ", ".join(str(x) for x in yrs_dict.keys()[-1:])

    # Training subset
    train = playerTable.loc[
        playerTable['mlb_season'].isin(yrs_dict.keys()[:-1]), :]

    # Test subset
    test = playerTable.loc[
        playerTable['mlb_season'].isin(yrs_dict.keys()[-1:]), :]

    # Save out the batter order to join post-prediction
    test_batter_order = test.loc[:, ['batter']]

    del playerTable
    gc.collect()

    # Fill NAs
    # Mean comes from subsetting table to only those obsv with
    # at least 1 missing value
    cols_for_repl = features + ['nextYearHits']
    train = fillna_process(train, col_list=cols_for_repl)
    test = fillna_process(test, col_list=cols_for_repl)

    # Final Data send out for checking before defining model
    train.to_csv(resultsPath + 'trainingDataFinal{}_{}.csv'.format(
        str(min(yrs_dict.keys())),
        str(max(yrs_dict.keys()))
    ))
    test.to_csv(resultsPath + 'testingDataFinal{}_{}.csv'.format(
        str(min(yrs_dict.keys())),
        str(max(yrs_dict.keys()))
    ))

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Prepare test results to be merged on after testing
    print "{} :: Defining randomForestRegressor model that will be " \
          "trained on prepped data".format(
        dt.datetime.now().strftime("%H:%M:%S")
    )
    regressor = RandomForestRegressor(
        n_estimators=12,
        max_features='auto'
    )

    print "{} :: Training randomForestRegressor model".format(
        dt.datetime.now().strftime("%H:%M:%S")
    )
    regressor.fit(
        train.as_matrix(features),
        train.as_matrix(['nextYearHits'])
    )

    print "{} :: Predicting \'nextYearHits\'".format(
        dt.datetime.now().strftime("%H:%M:%S")
    )
    pred = regressor.predict(
        test.as_matrix(features)
    )

    # Preparing prediction data before sending out
    test['predictedValue'] = pred
    test['batter'] = test_batter_order.loc[:, 'batter']
    test['residual'] = (
        test['predictedValue'] - test['nextYearHits']
    ).abs()
    test.to_csv(
        resultsPath + '{}-{}_predictedValuesComplete.csv'.format(
            str(min(yrs_dict.keys())),
            str(max(yrs_dict.keys()))
        )
    )



