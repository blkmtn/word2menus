import pandas as pd

# join function


def merge_data(df1: pd.DataFrame, df1_joinkey: str, df2: pd.DataFrame, 
               df2_joinkey: str, join_type: str, merged_dataset):
    merged_dataset = pd.merge(df1, df2, how = join_type, left_on = df1_joinkey,
                              right_on = df2_joinkey)
    return merged_dataset


