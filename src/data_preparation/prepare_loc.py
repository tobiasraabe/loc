import json

import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj
from sklearn.externals import joblib


# This list is ordered according to the item table in our paper.
PERCEIVED_CONTROL = [
    "LOC_LIFES_COURSE",
    "LOC_ACHIEVED_DESERVE",
    "LOC_LUCK",
    "LOC_OTHERS",
    "LOC_DOUBT",
    "LOC_POSSIBILITIES",
    "LOC_LITTLE_CONTROL",
]


LOC_VALUES = {
    "[1] Trifft ueberhaupt nicht zu": 1,
    "[2] [2/10]": 2,
    "[3] [3/10]": 3,
    "[4] [4/10]": 4,
    "[5] [5/10]": 5,
    "[6] [6/10]": 6,
    "[7] Trifft voll zu": 7,
}


def calculate_cronbachs_alpha(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems - 1.0) * (1 - itemvars.sum() / tscores.var(ddof=1))


def clean_variables(df):
    # Replace values and cast to integers
    for variable in df.select_dtypes("category"):
        df[variable].cat.rename_categories(LOC_VALUES, inplace=True)
        df[variable] = pd.to_numeric(df[variable], errors="raise", downcast="integer")
    return df


def invert_items(df):
    """This function inverts the scale of some items of LoC so that for all
    items higher numbers reflect greater feelings of control."""
    inverted_items = [
        "LOC_ACHIEVED_DESERVE",
        "LOC_LUCK",
        "LOC_OTHERS",
        "LOC_DOUBT",
        "LOC_POSSIBILITIES",
        "LOC_ABILITIES",
        "LOC_LITTLE_CONTROL",
    ]
    for item in inverted_items:
        df[item].replace(
            to_replace=[1, 2, 3, 4, 5, 6, 7], value=[7, 6, 5, 4, 3, 2, 1], inplace=True
        )

    return df


def create_index(df):
    """This function creates and index which is the average over all LoC
    items."""
    df["LOC_INDEX"] = df[PERCEIVED_CONTROL].mean(axis="columns")

    return df


if __name__ == "__main__":
    # Load dataset
    df = pd.read_pickle(ppj("OUT_DATA", "loc_raw.pkl"))
    # Clean the data
    df = clean_variables(df)
    # Invert items so that higher numbers indicate greater feelings of control
    df = invert_items(df)
    # Calculate Cronbach's alpha for the whole scale
    data = df[[i for i in df if "LOC" in i]].as_matrix().T
    cronbachs_alpha_ten = calculate_cronbachs_alpha(data)
    # Restrict to seven item scale proposed by Specht et al (2013)
    df = df[["ID", "YEAR"] + PERCEIVED_CONTROL]
    # Create an index as the average of LoC items
    df = create_index(df)
    # Calculate Cronbach's Alpha for seven item scale. First, reshape the data
    # to n (items) * p (observations)
    data = df[PERCEIVED_CONTROL].as_matrix().T
    cronbachs_alpha_seven = calculate_cronbachs_alpha(data)
    # Create container
    container = {}
    container["data"] = df
    # Save numbers to json
    with open(ppj("OUT_TABLES", "cronbachs_alphas.json"), "w") as file:
        file.write(
            json.dumps(
                {"ca_seven": cronbachs_alpha_seven, "ca_ten": cronbachs_alpha_ten}
            )
        )
    # Save data for PCA and FA
    joblib.dump(container, ppj("OUT_DATA", "loc_container.pkl"))
