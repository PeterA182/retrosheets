from __future__ import division

__author__ = 'paltamura'

import sys
import gc

import pandas as pd
import numpy as np
from string import zfill
import datetime as dt

from collections import OrderedDict

sys.path.append("E:\\Models\\retrosheets\\data\\")
from toolkit import compress_columns


#
# ---- ---- ----
# Methods


def add_columns(df, col_file_path):

    # Read in and convert to list
    read_in_cols = pd.read_csv(col_file_path)
    cols = list(read_in_cols['Description'])
    cols = [x.lower().replace(' ', '_').replace('-', '_') for x in cols]

    # Set as columns attribute of the table
    df.columns = cols


def add_event_definitions(df, event_file_path):

    # Read in to DataFrame to create dictionary form
    eventTypes = pd.read_csv(event_file_path)
    eventTypes = eventTypes.set_index('code')['event'].to_dict()

    # Map
    df['event_descr'] = df.loc[:, 'event_type'].apply(
        lambda x: eventTypes.get(x)
    )


def read_in_retrosheets(min_year, max_year, to_dict=False):
    """

    :param min_year:
    :param max_year:
    :param to_dict:
    :return:
    """

    # Establish dictionary if reading in to_dict
    if to_dict:
        perfYearsDict = OrderedDict()
    else:
        perfYearsTable = pd.DataFrame()

    # Establish range list of years to find files for
    for yr in range(min_year, max_year+1, 1):

        # Read in file
        perfYearCurr = pd.read_csv(
            retroSheetsPath + 'all{}.csv'.format(str(yr)),
            header=None
        )

        # Add columns to the DataFrame
        add_columns(perfYearCurr, col_file_path=cols_path)

        # Strip game_date from game_id
        perfYearCurr['game_date'] = (perfYearCurr['game_id'].str[3:11])
        perfYearCurr['game_date'] = pd.to_datetime(
            perfYearCurr['game_date'],
            format='%Y%m%d'
        )
        perfYearCurr['mlb_season'] = perfYearCurr['game_date'].dt.year

        # Determine location to send out to once prep complete
        if to_dict:
            perfYearsDict[str(yr)] = perfYearCurr
        else:
            perfYearsTable = perfYearsTable.append(perfYearCurr)

    # Determine what to return
    if to_dict:
        return perfYearsDict
    else:
        return perfYearsTable


def standardPrep(df, map_events=True, map_hit_value=True):
    """

    :param df:
    :return:
    """

    # Map events
    if map_events:
        add_event_definitions(df=df, event_file_path=events_path)

    # Flag hits
    msk = (df['hit_value'] > 0)
    df['hit_flag'] = np.where(msk, 1, 0)

    # Flag strikeouts
    msk = (df['event_descr'] == 'so')
    df['strikeout_flag'] = np.where(msk, 1, 0)

    # Map Hit Value
    if map_hit_value:
        hits_dict = {
            1: 'single',
            2: 'double',
            3: 'triple',
            4: 'home_run'
        }

        df['hit_type'] = df.loc[:, 'hit_value'].apply(
            lambda x: hits_dict.get(x)
        )

    # Reorder and sort to make game sequence more obvious between players
    id_cols = [
        'game_id', 'game_date', 'event_number', 'batter', 'pitcher',
        'pitch_sequence', 'pitcher_hand', 'hit_flag', 'strikeout_flag',
        'batted_ball_type', 'fielded_by', 'hit_location', 'event_number',
        'event_type', 'event_descr'
    ]
    details = [x for x in list(df.columns) if x not in id_cols]
    df = df.loc[:, id_cols + details]

    # Return it
    return df


def run_standard_analysis(df, level, outpath, min_hits):

    # Determine level column
    if level == 'player':
        level_col = 'batter'
    elif level == 'game':
        level_col = 'game_id'
    else:
        raise

    # Assign base metrics from hits
    # TODO does not currently adjust for innings ended when caught stealing
    df_grpd = df.groupby(['mlb_season', level_col], as_index=False).\
        agg({
        'hit_flag': [np.sum, np.mean, 'count']})
    df_grpd = compress_columns(table=df_grpd)

    # Rename out the metrics (will be name of func given list format in agg)
    df_grpd.rename(
        columns={
            'sum': 'hits',
            'mean': 'batting_average',
            'count': 'at_bats'
        },
        inplace=True
    )

    # Filter to min hits
    df_app = pd.DataFrame()
    for yr in [x for x in list(set(df_grpd['mlb_season']))]:
        msk = (
            (df_grpd['hits'] >= min_hits)
            &
            (df_grpd['mlb_season'] == yr)
        )
        df_temp = df_grpd.loc[msk, :]

        # Get percentiles
        df_temp.loc[:, 'cumsum'] = df_temp.loc[:, 'hits'].cumsum()
        df_temp.loc[:, 'cum_pct'] = (df_temp['cumsum'] / df_temp['hits'].sum())
        df_temp.loc[:, '25pctl'] = np.percentile(df_temp['hits'], q=25)
        df_temp.loc[:, '75pctl'] = np.percentile(df_temp['hits'], q=75)

        # Coefficeint of Variation
        df_temp.loc[:, 'cv'] = (
            df_temp['75pctl'] - df_temp['25pctl']
        )/df_temp.loc[:, 'hits'].median()

        # Append back
        df_app = df_app.append(df_temp)

    # Free up
    del df_temp
    gc.collect()

    # Send out
    df_app.to_csv(
        outpath +'{}_level_analysis.csv'.format(level)
    )


if __name__ == "__main__":

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---

    # ---- ----
    # Paths
    retroSheetsPath = "E:\\Data\\Retrosheets\\download.folder\\unzipped\\"
    projectPath = 'E:\\Models\\retrosheets\\data\\'
    cols_path = projectPath + 'resources\\column_listing.csv'
    events_path = projectPath + 'resources\\event_type_glossary.csv'

    # ---- ----
    # Vars
    min_hits = 40
    level = 'player'
    min_year = 2010
    max_year = 2016
    run_exploratory_analysis = False

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Read in all years and send to single output within project path
    df_all = read_in_retrosheets(min_year=min_year, max_year=max_year)
    df_all.to_csv(projectPath + 'df_all_{}-{}.csv'.format(
        str(min_year), str(max_year)
    ))

    # Prep the actual dataFrame now that raw format of combined years have
    # been saved
    df_all = standardPrep(df_all)

    # Save Sample
    sample = df_all.loc[df_all['game_id'] == 'ANA201403310', :]
    sample.to_csv(
        projectPath + "analysis\\sampleBattingData.csv"
    )

    # Save fully prepped data
    df_all.to_csv(
        projectPath + 'prepped\\fullBattingData_{}-{}.csv'.format(
            str(min_year), str(max_year)
        )
    )

    #
    # ---- ----
    # Run Exploratory Analysis
    if run_exploratory_analysis:
        run_standard_analysis(
            df=df_all, level=level, outpath=projectPath+'analysis\\',
            min_hits=min_hits
        )

