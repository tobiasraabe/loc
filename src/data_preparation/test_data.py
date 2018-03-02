#!/usr/bin/env python

import pytest
import pandas as pd

# Needed to run pytest within Waf and with tox
try:
    from project_paths import project_paths_join as ppj
except ImportError:
    from bld.project_paths import project_paths_join as ppj


@pytest.fixture(scope='module')
def df(request):
    # Read any dataframe and pass it to test functions
    dataframe = pd.Dataframe({'col_1': [1, 2, 3]})

    yield dataframe


def test_dataframe(df):
    assert df.col_1.sum() == 6
