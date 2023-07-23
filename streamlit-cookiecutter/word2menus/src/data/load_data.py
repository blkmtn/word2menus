# load required packages

import pandas as pd


# function to load host logs
def load_hostlog_data(lunch_metrics_file: str) -> pd.DataFrame:
    """_summary_

    Args:
        lunch_metrics_file (str): file containing lunch service metrics

    Returns:
        pd.DataFrame: _description_
    """
    hostlogs = pd.read_csv(lunch_metrics_file)
    return hostlogs


# function to load menus


def load_menu_data(menus_file: str) -> pd.DataFrame:
    """_summary_

    Args:
        menus_file (str): file containing menu data

    Returns:
        pd.DataFrame: _description_
    """
    menus = pd.read_csv(menus_file)

    return menus
