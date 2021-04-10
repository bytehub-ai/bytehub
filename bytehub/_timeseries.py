import pandas as pd
import functools

try:
    # Allow for a minimal install with no dask/pyarrow
    from dask import dataframe as dd
except ImportError:
    pass


def concat(dfs):
    """Concat dataframes for multiple features."""
    if len(dfs) == 0 or isinstance(dfs[0], pd.DataFrame):
        # Use Pandas concat
        return pd.concat(dfs, join="outer", axis=1).ffill()
    elif isinstance(dfs[0], dd.DataFrame):
        # Use Dask concat
        dfs = functools.reduce(
            lambda left, right: dd.merge(
                left, right, left_index=True, right_index=True, how="outer"
            ),
            [df for df in dfs],
        )
        return dfs.ffill()
    else:
        raise NotImplementedError(f"Unrecognised dataframe type: {type(dfs[0])}")


def transform(df, func):
    """Transform dataframe using function."""
    transformed = func(df)
    # Make sure output has a single column named 'value'
    if isinstance(transformed, pd.Series) or isinstance(transformed, dd.Series):
        transformed = transformed.to_frame("value")
    if isinstance(df, pd.DataFrame) and not isinstance(transformed, pd.DataFrame):
        raise RuntimeError(
            f"Transforms in this namespace should return Pandas dataframes or series"
        )
    if isinstance(df, dd.DataFrame) and not isinstance(transformed, dd.DataFrame):
        raise RuntimeError(
            f"Transforms in this namespace should return Dask dataframes or series"
        )
    if len(transformed.columns) != 1:
        raise RuntimeError(
            f"Transform function should return a dataframe with a datetime index and single value column"
        )
    transformed.columns = ["value"]
    return transformed
