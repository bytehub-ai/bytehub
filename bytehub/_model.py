import sqlalchemy as sa
from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy import Integer, String, Boolean, JSON, Enum
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
import pandas as pd
import dask
import dask.dataframe as dd
import numpy as np
import pyarrow as pa
import re
import copy
import types
import posixpath
import fsspec
import functools
from . import _utils as utils
from . import _connection as conn


Base = declarative_base()
valid_name = re.compile(r"^[a-zA-Z0-9\.#_-]+$")
partitions = Enum("year", "date", name="partition")


class FeatureStoreMixin(object):
    name = Column(String(128), primary_key=True, nullable=False)
    description = Column(String, default="")
    meta = Column(JSON, default={})
    version = Column(Integer, default=1, nullable=False)

    @validates("name")
    def _validate_name(self, key, value):
        if not valid_name.match(value):
            raise ValueError(f"Invalid name {value}")
        return value

    def as_dict(self):
        return {
            k: v if utils.is_jsonable(v) else str(v)
            for k, v in self.__dict__.items()
            if k[0] != "_"
        }

    def bump_version(self):
        if self.version:
            self.version += 1
        else:
            self.version = 1

    def update_from_dict(self, payload):
        if not payload:
            return
        if "name" in payload and self.name:
            raise ValueError(
                f"Cannot change name of {self.__class__.__name__}: use clone instead"
            )
        if "namespace" in payload and self.namespace:
            raise ValueError(
                f"Cannot change namespace of {self.__class__.__name__}: use clone instead"
            )
        for key, value in payload.items():
            if key == "meta" or key == "metadata":
                # See https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/
                self.meta = copy.deepcopy(self.meta)
                # Merge old metadata with new
                if self.meta:
                    self.meta.update(value)
                else:
                    self.meta = value
                # Remove any keys that no longer have values
                self.meta = {k: v for k, v in self.meta.items() if v is not None}
            else:
                # Update fields
                setattr(self, key, value) if hasattr(self, key) else None
        self.bump_version()


class Namespace(Base, FeatureStoreMixin):
    __tablename__ = "namespace"

    url = Column(String, nullable=False, unique=True)
    storage_options = Column(JSON, nullable=False, default={})

    @hybrid_property
    def namespace(self):
        return self.name

    @namespace.setter
    def namespace(self, value):
        self.name = value

    @namespace.expression
    def namespace(cls):
        return cls.name

    def clean(self):
        # Check for unused data and remove it
        fs, fs_token, paths = fsspec.get_fs_token_paths(
            self.url, storage_options=self.storage_options
        )
        feature_paths = fs.ls(posixpath.join(paths[0], "feature"))
        active_feature_names = [f.name for f in self.features]
        feature_data = [f.split("/")[-1] for f in feature_paths]
        for feature in feature_data:
            if feature not in active_feature_names:
                # Redundant data... delete it
                fs.rm(posixpath.join(paths[0], "feature", feature), recursive=True)


class Feature(Base, FeatureStoreMixin):
    __tablename__ = "feature"

    namespace = Column(String(128), ForeignKey("namespace.name"), primary_key=True)
    namespace_object = relationship("Namespace", backref="features")

    partition = Column(partitions, default="date", nullable=False)
    serialized = Column(Boolean, default=False, nullable=False)
    transform = Column(JSON, nullable=True)

    @hybrid_property
    def full_name(self):
        return f"{self.namespace}/{self.name}"

    @validates("serialized")
    def validate_serialized(self, key, value):
        if self.serialized is not None and value != self.serialized:
            raise ValueError("Cannot change serialized setting on existing feature")
        return value

    @validates("transform")
    def validate_transform(self, key, value):
        if not value:
            return value
        if not isinstance(value.get("function"), types.FunctionType):
            raise ValueError(
                "Transform must be a Python function, accepting a single dataframe input"
            )
        assert "function" in value.keys(), "Transform must have a function defined"
        assert "args" in value.keys(), "Transform must have arguments defined"
        # Convert function to base64/cloudpickle format
        return {
            "format": "cloudpickle",
            "function": utils.serialize(value["function"]),
            "args": value["args"],
        }

    @classmethod
    def clone_from(cls, other, namespace, name):
        if not isinstance(other, cls):
            raise ValueError(f"Must clone from another {cls.__name__}")
        clone = cls()
        # Build new Feature with same settings as old
        clone.namespace = namespace
        clone.name = name
        payload = other.as_dict()
        payload.pop("namespace")
        payload.pop("name")
        clone.update_from_dict(payload)
        return clone

    def apply_partition(self, dt, offset=0):
        if isinstance(dt, dd.core.Series):
            if self.partition == "year":
                return dt.dt.year + offset
            elif self.partition == "date":
                return (dt + pd.Timedelta(days=offset)).dt.date.to_string()
            else:
                raise NotImplementedError(f"{self.partition} has not been implemented")
        else:
            if self.partition == "year":
                return pd.Timestamp(dt).year + offset
            elif self.partition == "date":
                return str(pd.Timestamp(dt).date() + pd.Timedelta(days=offset))
            else:
                raise NotImplementedError(f"{self.partition} has not been implemented")

    def save(self, df):
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
                raise ValueError(
                    "Not sure whether to use timestamp index or time column"
                )
        # Check time column
        if "time" in ddf.columns:
            ddf = ddf.assign(time=ddf.time.astype("datetime64[ns]"))
            # Add partition column
            ddf = ddf.assign(partition=self.apply_partition(ddf.time))
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
        if self.serialized:
            ddf = ddf.map_partitions(
                lambda df: df.assign(value=df.value.apply(pd.io.json.dumps))
            )

        # Write to output location
        url = self.namespace_object.url
        storage_options = self.namespace_object.storage_options
        path = posixpath.join(url, "feature", self.name)
        # Build schema
        schema = {"time": pa.timestamp("ns"), "created_time": pa.timestamp("ns")}
        if self.partition == "year":
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
                storage_options=storage_options,
            )
        except Exception as e:
            raise RuntimeError(f"Unable to save data to {path}: {str(e)}")

    def load_transform(self, from_date, to_date, freq, time_travel, mode, callers=[]):
        # Get the SQLAlchemy session for this feature
        session = sa.inspect(self).session
        if not session:
            raise RuntimeError(f"{self.name} is not bound to an SQLAlchemy session")
        # Check for recursive transforms
        if self.full_name in callers:
            raise RuntimeError(
                f"Recursive feature transform detected on {self.full_name}"
            )
        # Load the transform function
        func = utils.deserialize(self.transform["function"])
        # Load the features to transform
        dfs = []
        # Load each requested feature
        for f in self.transform["args"]:
            namespace, name = f.split("/")[0], "/".join(f.split("/")[1:])
            feature = (
                session.query(Feature)
                .filter_by(name=name, namespace=namespace)
                .one_or_none()
            )
            if not feature:
                raise ValueError(f"No feature named {name} exists in {namespace}")
            # Load individual feature
            df = feature.load(
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                mode=mode,
                callers=[*callers, self.full_name],
            )
            dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        # Merge features into a single dataframe
        if mode == "pandas":
            dfs = pd.concat(dfs, join="outer", axis=1).ffill()
        elif mode == "dask":
            dfs = functools.reduce(
                lambda left, right: dd.merge(
                    left, right, left_index=True, right_index=True, how="outer"
                ),
                dfs,
            )
            dfs = dfs.ffill()
        else:
            raise NotImplementedError(f"{mode} has not been implemented")
        # Make sure columns are in the same order as args
        dfs = dfs[self.transform["args"]]
        # Apply transform function
        transformed = func(dfs)
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

    def load(
        self,
        from_date=None,
        to_date=None,
        freq=None,
        time_travel=None,
        mode="pandas",
        callers=[],
    ):
        # Does this feature need to be transformed?
        if self.transform:
            return self.load_transform(
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                mode=mode,
                callers=callers,
            )
        # Get location
        url = self.namespace_object.url
        storage_options = self.namespace_object.storage_options
        # Identify which partitions to read
        filters = []
        # TODO: Can this be achieved more efficiently with partitions?
        if from_date:
            filters.append(("time", ">=", pd.Timestamp(from_date)))
        if to_date:
            filters.append(("time", "<=", pd.Timestamp(to_date)))
        filters = [filters] if filters else None
        # Read the data
        path = posixpath.join(url, "feature", self.name)
        try:
            ddf = dd.read_parquet(
                path, engine="pyarrow", filters=filters, storage_options=storage_options
            )
            ddf = ddf.repartition(npartitions=ddf.npartitions)
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
        if self.serialized:
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
                samples = pd.DataFrame(
                    index=pd.date_range(from_date, to_date, freq=freq)
                )
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

    def last(self, mode="pandas"):
        # Fetch last feature value
        # TODO: Make this more efficient in Pandas mode
        ddf = self.load(mode=mode)
        result = ddf.tail(1)
        if result.empty:
            return None
        else:
            return result["value"].iloc[0]

    def delete_data(self):
        # Deletes all of the data on this feature
        fs, fs_token, paths = fsspec.get_fs_token_paths(
            self.namespace_object.url,
            storage_options=self.namespace_object.storage_options,
        )
        feature_path = posixpath.join(paths[0], "feature", self.name)
        try:
            fs.rm(feature_path, recursive=True)
        except FileNotFoundError:
            pass

    def import_data_from(self, other):
        # Copy data over from another feature
        if not isinstance(other, self.__class__):
            raise ValueError(f"Must clone from another {cls.__name__}")
        # Get location of other feature to copy from
        url = other.namespace_object.url
        storage_options = other.namespace_object.storage_options
        # Read the data
        path = posixpath.join(url, "feature", other.name)
        try:
            ddf = dd.read_parquet(
                path, engine="pyarrow", storage_options=storage_options
            )
            # Repartition to optimise files on new dataset
            ddf = ddf.repartition(partition_size="100MB")
        except Exception as e:
            # No data available
            return
        # Get location of this feature to copy to
        url = self.namespace_object.url
        storage_options = self.namespace_object.storage_options
        # Read the data
        path = posixpath.join(url, "feature", self.name)
        # Copy data to new location
        try:
            ddf.to_parquet(
                path,
                engine="pyarrow",
                compression="snappy",
                write_index=True,
                append=True,
                partition_on="partition",
                ignore_divisions=True,
                storage_options=storage_options,
            )
        except Exception as e:
            raise RuntimeError(f"Unable to save data to {path}: {str(e)}")
