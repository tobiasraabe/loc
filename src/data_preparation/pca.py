import numpy as np
import pandas as pd
from bld.project_paths import project_paths_join as ppj
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import LOC_PERCEIVED_CONTROL


def transformation(data, pca, loc_items):
    """Transform the data with the weights attained by the PCA fit method on
    the 2010 data. Append the two created axes to the dataset.
    """
    transformed_data = pca.transform(data[loc_items])
    data["FIRST_FACTOR"] = transformed_data[:, 0]
    data["SECOND_FACTOR"] = transformed_data[:, 1]

    return data


def main(df):
    # Separate data for the three different years. copy() is necessary to pass
    # a copy of the df and not only a reference to the original df.
    loc_2005 = df[df.YEAR == 2005].copy()
    loc_2010 = df[df.YEAR == 2010].copy()
    loc_2015 = df[df.YEAR == 2015].copy()

    # Fit the PCA on the 2010 data first.
    pca = Pipeline(steps=[("std", StandardScaler()), ("pca", PCA())])

    # Fit the PCA on the 2010 data
    pca.fit(loc_2010[LOC_PERCEIVED_CONTROL])
    # Save the model
    container = {"data": loc_2010, "model": pca}
    joblib.dump(container, ppj("OUT_DATA", "model_pca_ten_comp.pkl"))

    # As the variance of the first and second axes is larger than 1,
    # pick two axes.
    pca.set_params(pca__n_components=2)
    pca.fit(loc_2010[LOC_PERCEIVED_CONTROL])

    # Calculate loadings which are more similar to regression coefficients than
    # eigenvalues (https://stats.stackexchange.com/questions/143905)
    loadings = pca.named_steps["pca"].components_.T * np.sqrt(
        pca.named_steps["pca"].explained_variance_
    )
    df_loadings = pd.DataFrame(
        data=loadings,
        index=LOC_PERCEIVED_CONTROL,
        columns=["FIRST_FACTOR", "SECOND_FACTOR"],
    )
    df_loadings.to_pickle(ppj("OUT_DATA", "pca_loadings.pkl"))
    # Save the model
    container = {"data": loc_2010, "model": pca}
    joblib.dump(container, ppj("OUT_DATA", "model_pca_two_comp.pkl"))

    # Transform the three waves
    loc_data_2005 = transformation(loc_2005, pca, LOC_PERCEIVED_CONTROL)
    loc_data_2010 = transformation(loc_2010, pca, LOC_PERCEIVED_CONTROL)
    loc_data_2015 = transformation(loc_2015, pca, LOC_PERCEIVED_CONTROL)

    # Merge the three datasets with the two newly created axes.
    frames = [loc_data_2005, loc_data_2010, loc_data_2015]
    df = pd.concat(frames, axis="rows")

    # Calculate differences of first factor by substracting the previous from
    # the following period
    df.sort_values(["ID", "YEAR"], axis="rows", inplace=True)
    df["FIRST_FACTOR_DELTA"] = df.groupby("ID")["FIRST_FACTOR"].transform(
        pd.Series.diff
    )

    # Save the LoC variables with the two axes as a pickle file.
    df.to_pickle(ppj("OUT_DATA", "loc_pca.pkl"))


if __name__ == "__main__":
    # Load data from container
    df = joblib.load(ppj("OUT_DATA", "loc_container.pkl"))["data"]
    main(df)
