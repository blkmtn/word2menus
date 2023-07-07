#load required packages

import pandas as pd


#function to load host logs
def load_hostlog_data(lunch_metrics_file:str):
    hostlogs = pd.read_csv(lunch_metrics_file)
    return hostlogs

#function to load menus
def load_menu_data(menus_file:str):
    menus = pd.read_csv(menus_file)
    return menus