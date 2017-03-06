__author__ = 'paltamura'

import sys
import gc

import pandas as pd
import numpy as np


sys.path.append("E:\\Models\\retrosheets\\data\\")
from toolkit import compress_columns


#
# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----
# Methods


def slump_metrics(df, metrics_cols, threshholds):

    """
    compares season_to_date (STDT) average with hits, strikeouts to
    the rolling_10_games (rolling10games) average

    PARAMETERS
    ----------
    df: DataFrame
        DataFrame containing game level data for players
    thresh: float
        Percent that needs to be dipped below to get labeled slump

    """

    #
    # ---- ---- ----
    # Assertions
    assert type(df) == pd.DataFrame
    assert type(metrics_cols) == list
    assert type(threshholds) == list

    # Establish table with just merge key to merge metrics on to and return
    df_return = df.loc[:, ['mlb_season', 'batter']].\
        drop_duplicates(inplace=False)
    
    #
    # ---- ---- ----
    # Get season_to_date (STDT) average for metrics_cols
    for col in metrics_cols:

        # Get threshhold
        thresh = threshholds[metrics_cols.index(col)]

        # Perform initial groupby to get cumulative sum of current metric
        df_t = df.sort_values(
            by=['mlb_season', 'batter', 'game_date'],
            ascending=True,
            inplace=False
        )
        df_t = df_t.groupby(
            ['mlb_season', 'batter', 'game_date', 'game_id'],
            as_index=False).agg({col: np.sum})

        # INCREMENTAL MEAN
        # Create season_to_date metric for current metric
        df_t['{}_STDT'.format(col)] = (
            df_t[col].cumsum() / pd.Series(np.arange(1, len(df_t)+1), df.index)
        )

        # Flag where below threshold
        df_t['{}_STDT_{}pctl'.format(col, str(thresh))] = \
            df_t.groupby(['mlb_season', 'batter'])\
                ['{}_STDT'.format(col)].transform(
                lambda x: np.percentile(x, thresh)
            )

        # INITIAL FLAG
        # Flag games spent in slump
        df_t['in_{}_slump'.format(col)] = 0
        if col == 'hits':
            msk = (
                df_t['{}_STDT'.format(col)] <=
                df_t['{}_STDT_{}pctl'.format(col, str(thresh))]
            )
        elif col == 'strikeouts':
            msk = (
                df_t['{}_STDT'.format(col)] >=
                df_t['{}_STDT_{}pctl'.format(col, str(thresh))]
            )
        else:
            raise
        df_t.loc[msk, 'in_{}_slump'.format(col)] = 1

        # Create counter and increment (nstead of using integer index in mask)
        df_t['counter'] = 1
        df_t['increment_index'] = df_t.loc[:, 'counter'].cumsum()
        df_t.drop(labels=['counter'], axis=1, inplace=True)

        # FLAG FIRST INSTANCE OF SLUMP
        msk = (
            (df_t['in_{}_slump'.format(col)].shift(1) == 0)
            &
            (df_t['in_{}_slump'.format(col)] == 1)
        )
        df_t.loc[msk, '{}_slump_day_1'.format(col)] = df_t['increment_index']

        # if not equal to 0, match above
        msk = (
            (df_t['in_{}_slump'.format(col)] == 1)
            &
            (df_t['{}_slump_day_1'.format(col)].shift(1).notnull())
        )
        df_t.loc[msk, '{}_slump_day_1'.format(col)] = \
            df_t['{}_slump_day_1'.format(col)].shift(1)
        df_t.rename(
            columns={'{}_slump_day_1'.format(col): '{}_slump_id'.format(col)},
            inplace=True
        )

        # replace non-slump-days with null
        df_t.sort_values(
            by=['mlb_season', 'batter', 'game_date'],
            ascending=True,
            inplace=True)
        df_t.head(2000).to_csv(
            'E:\\slump_id_made_pregroupby{}.csv'.format(col)
        )

        # Prep table to add back on
        df_addback = df_t.groupby(
            ['mlb_season', 'batter'],
            as_index=False
        ).agg({'{}_slump_id'.format(col): [pd.Series.nunique, 'count']})
        df_addback = compress_columns(df_addback)
        df_addback.rename(
            columns={'nunique': '{}_slump_count'.format(col),
                     'count': '{}_slump_days'.format(col)},
            inplace=True
        )
        df_addback.loc[:, '{}_slump_avg_len'.format(col)] = (
            df_addback['{}_slump_days'.format(col)] /
            df_addback['{}_slump_count'.format(col)]
        )

        # Merge
        pre_len = len(df_return)
        df_return = pd.merge(
            df_return,
            df_addback,
            how='left',
            on=['mlb_season', 'batter']
        )
        if len(df_return) != pre_len:
            raise Exception("Merge of type \'left\' has added rows.")

        del df_addback
        gc.collect()

    return df_return


def streak_metrics(df, thresh=.85):
    """

    :return:
    """

    return ""


if __name__ == "__main__":
    pass

