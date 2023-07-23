import pandas as pd

# join function


def merge_data(
    df1: pd.DataFrame,
    df1_key: str,
    df2: pd.DataFrame,
    df2_key: str,
    join_type: str,
    merged_dataset,
) -> pd.DataFrame:
    """_summary_

    Args:
        df1 (pd.DataFrame): _description_
        df1_key (str): _description_
        df2 (pd.DataFrame): _description_
        df2_key (str): _description_
        join_type (str): _description_
        merged_dataset (_type_): _description_

    Returns:
        pd.DataFrame: _description_
    """

    merged_dataset = pd.merge(
        df1, df2, how=join_type, left_on=df1_key, right_on=df2_key
    )

    return merged_dataset


def drop_services_without_menus(data: pd.DataFrame,
                                subset_value: str) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): dataframe with potential duplicates
        subset_value (str): column to with missing values to filter out rows

    Returns:
        pd.DataFrame: data frame without missing values for specified column
    """

    filtered_data = data.dropna(subset=subset_value)

    return filtered_data


def filter_dropoff_services(
    data: pd.DataFrame, data_column: str, filter_value: str
) -> pd.DataFrame:
    """_summary_

    Args:
        data (pd.DataFrame): dataframe to filter
        data_column (str): column to filter on
        filter_value (str): value to filter by

    Returns:
        pd.DataFrame: subset of original data filtered by input params
    """

    filtered_data = data[data[data_column] == filter_value]

    return filtered_data


def remove_blank_rows(
    data: pd.DataFrame, axis_value: int, how_value: str
) -> pd.DataFrame:
    """drops rows with missing values from dataframe

    Args:
        data (pd.DataFrame): input dataframe with missing values
        axis_value (int): pandas arg
        how_value (str): pandas arg

    Returns:
        pd.DataFrame: returns data frame without missing values
    """
    filtered_data = data.dropna(axis=axis_value, how=how_value)

    return filtered_data


def subset_column_as_df(data: pd.DataFrame,
                        subset_column: str) -> pd.Dataframe:
    """creates a single column dataframe subset from a larger dataframe
       for use in NLP transformations

    Args:
        data (pd.DataFrame): full input data
        subset_column (str): column in dataframe

    Returns:
        pd.Dataframe: single column dataframe
    """
    single_column_df = data = data[[subset_column]]

    return single_column_df
