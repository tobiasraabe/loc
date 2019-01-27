import pandas as pd
from bld.project_paths import project_paths_join as ppj
from sklearn.decomposition import FactorAnalysis
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline

from src import LOC_PERCEIVED_CONTROL


def transformation(data, fa, loc_items):
    """Transform the data with the weights attained by the FactorAnalysis fit
    method on the 2010 data. Append the two created axes to the dataset.
    """
    transformed_data = fa.transform(data[loc_items])
    data["FIRST_FACTOR"] = transformed_data[:, 0]
    data["SECOND_FACTOR"] = transformed_data[:, 1]

    return data


def main(df):
    # Separate data for the three different years. copy() is necessary to pass
    # a copy of the df and not only a reference to the original df.
    loc_2005 = df[df.YEAR == 2005].copy()
    loc_2010 = df[df.YEAR == 2010].copy()
    loc_2015 = df[df.YEAR == 2015].copy()

    # Fit the FactorAnalysis on the 2010 data first.
    fa = Pipeline(steps=[("fa", FactorAnalysis(max_iter=10000, svd_method="lapack"))])

    # Fit the FactorAnalysis on the 2010 data
    fa.fit(loc_2010[LOC_PERCEIVED_CONTROL])
    # Save the model
    container = {"data": loc_2010, "model": fa}
    joblib.dump(container, ppj("OUT_DATA", "model_fa_ten_comp.pkl"))

    # As the variance of the first and second axes is larger than 1,
    # pick two axes.
    fa.set_params(fa__n_components=2)
    fa.fit(loc_2010[LOC_PERCEIVED_CONTROL])
    # Save the loadings
    loadings = fa.named_steps["fa"].components_.T
    df_loadings = pd.DataFrame(
        data=loadings,
        index=LOC_PERCEIVED_CONTROL,
        columns=["FIRST_FACTOR", "SECOND_FACTOR"],
    )
    df_loadings.to_pickle(ppj("OUT_DATA", "fa_loadings.pkl"))

    # Save the model
    container = {"data": loc_2010, "model": fa}
    joblib.dump(container, ppj("OUT_DATA", "model_fa_two_comp.pkl"))
    # Transform the three waves
    loc_data_2005 = transformation(loc_2005, fa, LOC_PERCEIVED_CONTROL)
    loc_data_2010 = transformation(loc_2010, fa, LOC_PERCEIVED_CONTROL)
    loc_data_2015 = transformation(loc_2015, fa, LOC_PERCEIVED_CONTROL)

    # Merge the three datasets with the two newly created axes.
    frames = [loc_data_2005, loc_data_2010, loc_data_2015]
    df = pd.concat(frames, axis="rows")

    # Calculate differences of first factor by substracting the previous from
    # the following period
    df.sort_values(["ID", "YEAR"], axis="rows", inplace=True)
    df["FIRST_FACTOR_DELTA"] = df.groupby("ID")["FIRST_FACTOR"].transform(
        pd.Series.diff
    )

    # Test that the highest value in the LoC scores is connected to the highest
    # score on the first factor. This means that higher first factor indicate
    # more internal LoC.
    assert (
        df.loc[df.LOC_INDEX == df.LOC_INDEX.max(), "FIRST_FACTOR"]
        == df.FIRST_FACTOR.max()
    ).all(), "LoC is reversed!"

    # Save the LoC variables with the two axes as a pickle file.
    df.to_pickle(ppj("OUT_DATA", "loc_fa.pkl"))


if __name__ == "__main__":
    # Load data from container
    df = joblib.load(ppj("OUT_DATA", "loc_container.pkl"))["data"]
    main(df)
