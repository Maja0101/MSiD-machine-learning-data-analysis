import pandas as pd
import numpy as np  

def prepare_data(more_data=False):
    data_set = pd.read_csv('AviationData.csv', encoding='ISO-8859-1', low_memory=False)

    selected_cols = ["Aircraft.damage", "Total.Fatal.Injuries", "Total.Serious.Injuries", "Total.Minor.Injuries", "Total.Uninjured", "Number.of.Engines"]

    data_set = data_set[selected_cols]

    data_set.columns = data_set.columns.str.replace(".", "_")

    data_set['Total_Fatal_Injuries'] = data_set['Total_Fatal_Injuries'].fillna(0)
    data_set['Total_Serious_Injuries'] = data_set['Total_Serious_Injuries'].fillna(0)
    data_set['Total_Minor_Injuries'] = data_set['Total_Minor_Injuries'].fillna(0)
    data_set['Total_Uninjured'] = data_set['Total_Uninjured'].fillna(0)

    data_set['Number_of_Engines'] = data_set['Number_of_Engines'].fillna(0)

    data_set['Aircraft_damage'] = data_set['Aircraft_damage'].map({
        'Substantial':2,
        'Destroyed':3,
        'Minor':1,
        'Unknown':0})
    
    data_set['Aircraft_damage'] = data_set['Aircraft_damage'].fillna(0)

    X = []
    y = []

    for row in data_set.itertuples():
        souls = row.Total_Fatal_Injuries + row.Total_Serious_Injuries + row.Total_Minor_Injuries + row.Total_Uninjured
        if souls != 0:
            X.append(row.Total_Fatal_Injuries / souls)
        else:
            X.append(0)

    X = np.array(X).reshape(-1, 1)
    y = data_set['Aircraft_damage'].values.astype(int)

    if more_data:
        sel_X = data_set[["Total_Fatal_Injuries", "Total_Serious_Injuries", "Total_Minor_Injuries", "Total_Uninjured", "Number_of_Engines"]]
        X = np.hstack((sel_X.values, X))
    
    return X, y
