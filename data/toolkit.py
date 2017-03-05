__author__ = 'paltamura'

#
# ---- ---- ----
# Methods


def compress_columns(table, keep_level=1, combine=False, reset_index=False):

    # If reset_index
    if reset_index:
        table = table.reset_index(drop=False)

    # If combine - then construct combined columns
    if combine:
        col_new = (
            lambda x: x[0] if x[keep_level] == '' else x[0] + '_' + x[1]
        )
        table.columns = [col_new(col) for col in table.columns]

    else:

        # Keep highest level until blank below
        col_new = (
            lambda x: x[0] if x[keep_level] == '' else x[keep_level]
        )
        table.columns = [col_new(col) for col in table.columns]

    return table


if __name__ == "__main__":
    pass