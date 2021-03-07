import pandas as pd
import posixpath
import functools
import numpy as np

try:
    # Allow for a minimal install with no dask/pyarrow
    from dask import dataframe as dd
    import dask
    import pyarrow as pa
    import fsspec
except ImportError:
    pass


def _clean_dict(d):
    """Cleans dictionary of extraneous keywords."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


def delete(name, url, storage_options):
    """Delete timeseries data for a feature."""
    fs, fs_token, paths = fsspec.get_fs_token_paths(
        url,
        storage_options=_clean_dict(storage_options),
    )
    feature_path = posixpath.join(paths[0], "feature", name)
    try:
        fs.rm(feature_path, recursive=True)
    except FileNotFoundError:
        pass


def load(
    name,
    url,
    storage_options,
    from_date,
    to_date,
    freq,
    time_travel,
    mode,
    serialized,
    partitions=None,
):
    """Load timeseries data from storage."""
    # Identify which partitions to read
    filters = []
    if from_date:
        filters.append(("time", ">=", pd.Timestamp(from_date)))
    if to_date:
        filters.append(("time", "<=", pd.Timestamp(to_date)))
    if partitions:
        for p in partitions:
            filters.append(("partition", "==", p))
    filters = [filters] if filters else None
    # Read the data
    path = posixpath.join(url, "feature", name)
    try:
        ddf = dd.read_parquet(
            path,
            engine="pyarrow",
            filters=filters,
            storage_options=_clean_dict(storage_options),
        )
        ddf = ddf.repartition(npartitions=ddf.npartitions)
    except PermissionError as e:
        raise e
    except Exception as e:
        # No data available
        empty_df = pd.DataFrame(
            columns=["time", "created_time", "value", "partition"]
        ).set_index("time")
        ddf = dd.from_pandas(empty_df, chunksize=1)
    # Apply time-travel
    if time_travel:
        ddf = ddf.reset_index()
        ddf = ddf[ddf.created_time <= ddf.time + pd.Timedelta(time_travel)]
        ddf = ddf.set_index("time")
    # De-serialize from JSON if required
    if serialized:
        ddf = ddf.map_partitions(
            lambda df: df.assign(value=df.value.apply(pd.io.json.loads)),
            meta={
                "value": "object",
                "created_time": "datetime64[ns]",
                "partition": "uint16",
            },
        )
    if not from_date:
        from_date = ddf.index.min().compute()  # First value in data
    if not to_date:
        to_date = ddf.index.max().compute()  # Last value in data
    if mode == "pandas":
        # Convert to Pandas
        pdf = ddf.compute()
        # Keep only last created_time for each index timestamp
        pdf = (
            pdf.reset_index()
            .set_index("created_time")
            .sort_index()
            .groupby("time")
            .last()
        )
        # Apply resampling/date filtering
        if freq:
            samples = pd.DataFrame(index=pd.date_range(from_date, to_date, freq=freq))
            pdf = pd.merge(
                pd.merge(
                    pdf, samples, left_index=True, right_index=True, how="outer"
                ).ffill(),
                samples,
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:
            # Filter on date range
            pdf = pdf.loc[pd.Timestamp(from_date) : pd.Timestamp(to_date)]
        return pdf.drop(columns="partition")

    elif mode == "dask":
        # Keep only last created_time for each index timestamp
        delayed_apply = dask.delayed(
            # Use pandas on each dask partition
            lambda x: x.reset_index()
            .set_index("created_time")
            .sort_index()
            .groupby("time")
            .last()
        )
        ddf = dd.from_delayed([delayed_apply(d) for d in ddf.to_delayed()])
        # Apply resampling/date filtering
        if freq:
            # Index samples for final dataframe
            samples = dd.from_pandas(
                pd.DataFrame(index=pd.date_range(from_date, to_date, freq=freq)),
                chunksize=100000,
            )
            ddf = dd.merge(
                # Interpolate
                dd.merge(
                    ddf, samples, left_index=True, right_index=True, how="outer"
                ).ffill(),
                samples,
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:
            # Filter on date range
            ddf = ddf.loc[pd.Timestamp(from_date) : pd.Timestamp(to_date)]
        return ddf.drop(columns="partition")
    else:
        raise ValueError(f'Unknown mode: {mode}, should be "pandas" or "dask"')


def apply_partition(partition, dt, offset=0):
    if isinstance(dt, dd.core.Series):
        if partition == "year":
            return dt.dt.year + offset
        elif partition == "date":
            return (dt + pd.Timedelta(days=offset)).dt.date.to_string()
        else:
            raise NotImplementedError(f"{partition} has not been implemented")
    else:
        if partition == "year":
            return pd.Timestamp(dt).year + offset
        elif partition == "date":
            return str(pd.Timestamp(dt).date() + pd.Timedelta(days=offset))
        else:
            raise NotImplementedError(f"{partition} has not been implemented")


def list_partitions(name, url, storage_options, n=None, reverse=False):
    """List the available partitions for a feature."""
    fs, fs_token, paths = fsspec.get_fs_token_paths(
        url,
        storage_options=_clean_dict(storage_options),
    )
    feature_path = posixpath.join(paths[0], "feature", name)
    try:
        objects = fs.ls(feature_path)
    except FileNotFoundError:
        return []
    partitions = [obj for obj in objects if obj.startswith("partition=")]
    partitions = [p.split("=")[1] for p in partitions]
    partitions = sorted(partitions, reverse=reverse)
    if n:
        partitions = partitions[:n]
    return partitions


def save(df, name, url, storage_options, partition="date", serialized=False):
    """Save dataframe to storage location."""
    if df.empty:
        # Nothing to do
        return
    # Convert Pandas -> Dask
    if isinstance(df, pd.DataFrame):
        ddf = dd.from_pandas(df, chunksize=100000)
    elif isinstance(df, dd.DataFrame):
        ddf = df
    else:
        raise ValueError("Data must be supplied as a Pandas or Dask DataFrame")
    # Check value columm
    if "value" not in ddf.columns:
        raise ValueError("DataFrame must contain a value column")
    # Check we have a timestamp index column
    if np.issubdtype(ddf.index.dtype, np.datetime64):
        ddf = ddf.reset_index()
        if "time" in df.columns:
            raise ValueError("Not sure whether to use timestamp index or time column")
    # Check time column
    if "time" in ddf.columns:
        ddf = ddf.assign(time=ddf.time.astype("datetime64[ns]"))
        # Add partition column
        ddf = ddf.assign(partition=apply_partition(partition, ddf.time))
        ddf = ddf.set_index("time")
    else:
        raise ValueError(
            f"DataFrame must be supplied with timestamps, not {ddf.index.dtype}"
        )
    # Check for created_time column
    if "created_time" not in ddf.columns:
        ddf = ddf.assign(created_time=pd.Timestamp.now())
    else:
        ddf = ddf.assign(created_time=ddf.created_time.astype("datetime64[ns]"))
    # Check for extraneous columns
    extraneous = set(ddf.columns) - set(["created_time", "value", "partition"])
    if len(extraneous) > 0:
        raise ValueError(f"DataFrame contains extraneous columns: {extraneous}")
    # Serialize to JSON if required
    if serialized:
        ddf = ddf.map_partitions(
            lambda df: df.assign(value=df.value.apply(pd.io.json.dumps))
        )

    # Write to output location
    path = posixpath.join(url, "feature", name)
    # Build schema
    schema = {"time": pa.timestamp("ns"), "created_time": pa.timestamp("ns")}
    if partition == "year":
        schema["partition"] = pa.uint16()
    else:
        schema["partition"] = pa.string()
    for field in pa.Table.from_pandas(ddf.head()).schema:
        if field.name == "value":
            schema["value"] = field.type
    try:
        ddf.to_parquet(
            path,
            engine="pyarrow",
            compression="snappy",
            write_index=True,
            append=True,
            partition_on="partition",
            ignore_divisions=True,
            schema=schema,
            storage_options=_clean_dict(storage_options),
        )
    except Exception as e:
        raise RuntimeError(f"Unable to save data to {path}: {str(e)}")


def copy(
    from_name, from_url, from_storage_options, to_name, to_url, to_storage_options
):
    """Used during clone operations to copy timeseries data between locations."""
    # Read the data
    path = posixpath.join(from_url, "feature", from_name)
    try:
        ddf = dd.read_parquet(
            path, engine="pyarrow", storage_options=_clean_dict(from_storage_options)
        )
        # Repartition to optimise files on new dataset
        ddf = ddf.repartition(partition_size="100MB")
    except Exception as e:
        # No data available
        return
    # Copy data to new location
    path = posixpath.join(to_url, "feature", to_name)
    ddf.to_parquet(
        path,
        engine="pyarrow",
        compression="snappy",
        write_index=True,
        append=True,
        partition_on="partition",
        ignore_divisions=True,
        storage_options=_clean_dict(to_storage_options),
    )


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
            dfs,
        )
        return dfs.ffill()
    else:
        raise NotImplementedError(f"Unrecognised dataframe type: {type(dfs[0])}")


def feature_paths(url, storage_options):
    """List storage paths for features."""
    fs, fs_token, paths = fsspec.get_fs_token_paths(
        url,
        storage_options=_clean_dict(storage_options),
    )
    feature_paths = fs.ls(posixpath.join(paths[0], "feature"))
    return feature_paths


def transform(df, func, mode):
    """Transform dataframe using function."""
    transformed = func(df)
    # Make sure output has a single column named 'value'
    if isinstance(transformed, pd.Series) or isinstance(transformed, dd.Series):
        transformed = transformed.to_frame("value")
    if mode == "pandas" and not isinstance(transformed, pd.DataFrame):
        raise RuntimeError(
            f"This featurestore has a Pandas backend, therefore transforms should return Pandas dataframes"
        )
    if mode == "dask" and not isinstance(transformed, pd.DataFrame):
        raise RuntimeError(
            f"This featurestore has a Dask backend, therefore transforms should return Dask dataframes"
        )
    if len(transformed.columns) != 1:
        raise RuntimeError(
            f"Transform function should return a dataframe with a datetime index and single value column"
        )
    transformed.columns = ["value"]
    return transformed
