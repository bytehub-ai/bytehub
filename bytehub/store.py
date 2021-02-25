import sqlalchemy as sa
import pandas as pd
from dask import dataframe as dd
import posixpath
import functools
import json
from sqlalchemy.sql import text
from . import _connection as conn
from . import _model as model


class FeatureStore:
    """ByteHub Feature Store object"""

    def __init__(self, connection_string="sqlite:///bytehub.db", backend="pandas"):
        """Create a Feature Store, or connect to an existing one

        Args:
            connection_string, str: SQLAlchemy connection string for database
                containing feature store metadata - defaults to local sqlite file
            backend, str: eith 'pandas' (default) or 'dask', specifying the type
                of dataframes returned by load_dataframe
        """
        self.engine, self.session_maker = conn.connect(connection_string)
        if backend.lower() not in ["pandas", "dask"]:
            raise ValueError("Backend must be either pandas or dask")
        self.mode = backend.lower()
        model.Base.metadata.create_all(self.engine)

    @classmethod
    def _split_name(cls, namespace=None, name=None):
        """Parse namespace and name."""
        if not namespace and name and "/" in name:
            parts = name.split("/")
            namespace, name = parts[0], "/".join(parts[1:])
        return namespace, name

    @classmethod
    def _validate_kwargs(cls, args, valid=[], mandatory=[]):
        for name in args.keys():
            if name not in valid:
                raise ValueError(f"Invalid argument: {name}")
        for name in mandatory:
            if name not in args.keys():
                raise ValueError(f"Missing mandatory argument: {name}")

    @classmethod
    def _unpack_list(cls, obj, namespace=None):
        """Extract namespace, name combinations from DataFrame or list
        and return as list of tuples
        """
        if isinstance(obj, str):
            return [FeatureStore._split_name(namespace=namespace, name=obj)]
        elif isinstance(obj, pd.DataFrame):
            # DataFrame format must have a name column
            df = obj
            if "name" not in df.columns:
                raise ValueError("DataFrame must have a name column")
            return [
                (row.get("namespace", namespace), row.get("name"))
                for _, row in df.iterrows()
            ]
        elif isinstance(obj, list):
            # Could be list of names, of list of dictionaries
            r = []
            for item in obj:
                if isinstance(item, str):
                    r.append(FeatureStore._split_name(name=item, namespace=namespace))
                elif isinstance(item, dict):
                    r.append(
                        FeatureStore._split_name(
                            namespace=item.get("namespace"), name=item.get("name")
                        )
                    )
                else:
                    raise ValueError("List must contain strings or dicts")
            return r
        else:
            raise ValueError(
                "Must supply a string, dataframe or list specifying namespace/name"
            )

    def _list(self, cls, namespace=None, name=None, regex=None):
        namespace, name = FeatureStore._split_name(namespace, name)
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            # Filter by namespace
            if namespace:
                r = r.filter_by(namespace=namespace)
            # Filter by matching name
            if name:
                r = r.filter_by(name=name)
            objects = r.all()
            df = pd.DataFrame([obj.as_dict() for obj in objects])
            if df.empty:
                return pd.DataFrame()
            # Filter by regex search on name
            if regex:
                df = df[df.name.str.contains(regex)]
            # List transforms as simple true/false
            if "transform" in df.columns:
                df = df.assign(transform=df["transform"].apply(lambda x: x is not None))
            # Sort the columns
            column_order = ["namespace", "name", "version", "description", "meta"]
            column_order = [c for c in column_order if c in df.columns]
            df = df[[*column_order, *df.columns.difference(column_order)]]
            return df

    def _delete(self, cls, namespace=None, name=None, delete_data=False):
        namespace, name = FeatureStore._split_name(namespace, name)
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise RuntimeError(
                    f"No existing {cls.__name__} named {name} in {namespace}"
                )
            if hasattr(obj, "delete_data") and delete_data:
                obj.delete_data()
            session.delete(obj)

    def _update(self, cls, namespace=None, name=None, payload={}):
        namespace, name = FeatureStore._split_name(namespace, name)
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise RuntimeError(
                    f"No existing {cls.__name__} named {name} in {namespace}"
                )
            # Apply updates from payload
            obj.update_from_dict(payload)

    def _create(self, cls, namespace=None, name=None, payload={}):
        if cls is model.Namespace:
            payload.update({"name": name})
        else:
            namespace, name = FeatureStore._split_name(namespace, name)
            if not self._exists(model.Namespace, namespace=namespace):
                raise ValueError(f"{namespace} namespace does not exist")
            payload.update({"name": name, "namespace": namespace})
        with conn.session_scope(self.session_maker) as session:
            obj = cls()
            obj.update_from_dict(payload)
            session.add(obj)

    def _exists(self, cls, namespace=None, name=None):
        ls = self._list(cls, namespace=namespace, name=name)
        return not ls.empty

    def list_namespaces(self, **kwargs):
        """List namespaces in the feature store.

        Search by name or regex query.

        Args:
            name, str, optional: name of namespace to filter by.
            namespace, str, optional: same as name.
            regex, str, optional: regex filter on name.

        Returns:
            pd.DataFrame: DataFrame of namespaces and metadata.
        """

        self.__class__._validate_kwargs(kwargs, ["name", "namespace", "regex"])
        return self._list(
            model.Namespace,
            name=kwargs.get("name", kwargs.get("namespace")),
            regex=kwargs.get("regex"),
        )

    def create_namespace(self, name, **kwargs):
        """Create a new namespace in the feature store.

        Args:
            name, str: name of the namespace
            description, str, optional: description for this namespace
            url, str: url of data store
            storage_options, dict, optional: storage options to be passed to Dask
            meta, dict, optional: key/value pairs of metadata
        """
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "url", "storage_options", "meta"],
            mandatory=["url"],
        )
        self._create(model.Namespace, name=name, payload=kwargs)

    def update_namespace(self, name, **kwargs):
        """Update a namespace in the feature store.

        Args:
            name, str: namespace to update
            description, str, optional: updated description
            storage_options, dict, optional: updated storage_options
            meta, dict, optional: updated key/value pairs of metadata
        """
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "storage_options", "meta"],
        )
        self._update(model.Namespace, name=name, payload=kwargs)

    def delete_namespace(self, name):
        """Delete a namespace from the feature store.

        Args:
            name: namespace to be deleted.
        """
        if not self.list_features(namespace=name).empty:
            raise RuntimeError(
                f"{name} still contains features: these must be deleted first"
            )
        self._delete(model.Namespace, name=name)

    def clean_namespace(self, name):
        """Removes any data that is not associated with features in the namespace.
        Run this to free up disk space after deleting features

        Args:
            name:, str: namespace to clean
        """
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            namespace = r.one_or_none()
            if not namespace:
                raise RuntimeError(f"No existing Namespace named {name}")
            namespace.clean()

    def list_features(self, **kwargs):
        """List features in the feature store.

        Search by namespace, name and/or regex query

        Args:
            name, str, optional: name of feature to filter by.
            namespace, str, optional: namespace to filter by.
            regex, str, optional: regex filter on name.

        Returns:
            pd.DataFrame: DataFrame of features and metadata.
        """

        FeatureStore._validate_kwargs(kwargs, valid=["name", "namespace", "regex"])
        return self._list(
            model.Feature,
            namespace=kwargs.get("namespace"),
            name=kwargs.get("name"),
            regex=kwargs.get("regex"),
        )

    def create_feature(self, name, namespace=None, **kwargs):
        """Create a new feature in the feature store.

        Args:
            name, str: name of the feature
            namespace, str, optional: namespace which should hold this feature
            description, str, optional: description for this namespace
            partition, str, optional: partitioning of stored timeseries (default: 'date')
            serialized, bool, optional: if True, converts values to JSON strings before saving,
                which can help in situations where the format/schema of the data changes
                over time
            transform: str, optional: pickled function code for feature transforms
            meta, dict, optional: key/value pairs of metadata
        """
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "partition", "serialized", "transform"],
            mandatory=[],
        )
        self._create(model.Feature, namespace=namespace, name=name, payload=kwargs)

    def clone_feature(self, name, namespace=None, **kwargs):
        """Create a new feature by cloning an existing one.

        Args:
            name, str: name of the feature
            namespace, str, optional: namespace which should hold this feature
            from_name, str: the name of the existing feature to copy from
            from_namespace, str, optional: namespace of the existing feature
        """
        self.__class__._validate_kwargs(
            kwargs,
            valid=["from_namespace", "from_name"],
            mandatory=["from_name"],
        )
        from_namespace, from_name = FeatureStore._split_name(
            kwargs.get("from_namespace"), kwargs.get("from_name")
        )
        to_namespace, to_name = FeatureStore._split_name(namespace, name)
        if not self._exists(model.Namespace, namespace=from_namespace):
            raise ValueError(f"{from_namespace} namespace does not exist")
        if not self._exists(model.Namespace, namespace=to_namespace):
            raise ValueError(f"{to_namespace} namespace does not exist")
        with conn.session_scope(self.session_maker) as session:
            # Get the existing feature
            r = session.query(model.Feature)
            r = r.filter_by(namespace=from_namespace, name=from_name)
            feature = r.one_or_none()
            if not feature:
                raise RuntimeError(
                    f"No existing Feature named {from_name} in {from_namespace}"
                )
            # Create the new feature
            new_feature = model.Feature.clone_from(feature, to_namespace, to_name)
            session.add(new_feature)
            # Copy data to new feature, if this raises exception will rollback
            if not new_feature.transform:
                new_feature.import_data_from(feature)

    def delete_feature(self, name, namespace=None, delete_data=False):
        """Delete a feature from the feature store.

        Args:
            name, str: name of feature to delete.
            namespace, str: namespace, if not included in feature name.
            delete_data, boolean, optional: if set to true will delete underlying stored data
                for this feature, otherwise default behaviour is to delete the feature store
                metadata but leave the stored timeseries values intact
        """
        self._delete(model.Feature, namespace, name, delete_data=delete_data)

    def update_feature(self, name, namespace=None, **kwargs):
        """Update a namespace in the feature store.

        Args:
            name, str: feature to update
            namespace, str: namespace, if not included in feature name
            description, str, optional: updated description
            transform: str, optional: pickled function code for feature transforms
            meta, dict, optional: updated key/value pairs of metadata
        """
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "transform"],
        )
        self._update(model.Feature, name=name, namespace=namespace, payload=kwargs)

    def transform(self, name, namespace=None, from_features=[]):
        """Decorator for creating/updating virtual (transformed) features.
        Use this on a function that accepts a dataframe input and returns an output dataframe
        of tranformed values.

        Args:
            name, str: feature to update
            namespace, str: namespace, if not included in feature name
            from_features, list[str]: list of features which should be transformed by this one
        """

        def decorator(func):
            # Create or update feature with transform
            to_namespace, to_name = self._split_name(namespace=namespace, name=name)
            computed_from = [f"{ns}/{n}" for ns, n in self._unpack_list(from_features)]
            for feature in computed_from:
                assert self._exists(
                    model.Feature, name=feature
                ), f"{feature} does not exist in the feature store"

            transform = {"function": func, "args": computed_from}
            payload = {"transform": transform, "description": func.__doc__}
            if self._exists(model.Feature, namespace=to_namespace, name=to_name):
                # Already exists, update it
                self.update_feature(to_name, namespace=to_namespace, **payload)
            else:
                # Create a new feature
                self.create_feature(to_name, namespace=to_namespace, **payload)
            # Call the transform
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func

        return decorator

    def load_dataframe(
        self,
        features,
        from_date=None,
        to_date=None,
        freq=None,
        time_travel=None,
    ):
        """Load a DataFrame of feature values from the feature store.

        Args:
            features, str, list, pd.DataFrame: name of feature to load, or list/DataFrame of feature namespaces/name
            from_date, datetime, optional: start date to load timeseries from, defaults to everything
            to_date, datetime, optional: end date to load timeseries to, defaults to everything
            freq, str, optional: frequency interval at which feature values should be sampled
            time_travel, str, optional: timedelta string, indicating that time-travel should be applied to the
                returned timeseries values, useful in forecasting applications

        Returns:
            pd.DataFrame or dask.DataFrame depending on which backend was specified in the feature store
        """
        dfs = []
        # Load each requested feature
        for f in self._unpack_list(features):
            namespace, name = f
            with conn.session_scope(self.session_maker) as session:
                feature = (
                    session.query(model.Feature)
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
                    mode=self.mode,
                )
                dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        if self.mode == "pandas":
            return pd.concat(dfs, join="outer", axis=1).ffill()
        elif self.mode == "dask":
            dfs = functools.reduce(
                lambda left, right: dd.merge(
                    left, right, left_index=True, right_index=True, how="outer"
                ),
                dfs,
            )
            return dfs.ffill()
        else:
            raise NotImplementedError(f"{self.mode} has not been implemented")

    def save_dataframe(self, df, name=None, namespace=None):
        """Save a DataFrame of feature values to the feature store.

        Args:
            df, pd.DataFrame: DataFrame of feature values
                Must have a 'time' column or DateTimeIndex of time values
                Optionally include a 'created_time' (defaults to utcnow() if omitted)
                For a single feature a 'value' column, or column header of feature namespace/name
                For multiply features name the columns using namespace/name
            name, str, optional: Name of feature, if not included in DataFrame column name
            namespace, str, optional: Namespace, if not included in DataFrame column name
        """
        # Check dataframe columns
        feature_columns = df.columns.difference(["time", "created_time"])
        if len(feature_columns) == 1:
            # Single feature to save
            if feature_columns[0] == "value":
                if not name:
                    raise ValueError("Must specify feature name")
            else:
                name = feature_columns[0]
                df = df.rename(columns={name: "value"})
            if not self._exists(model.Feature, namespace, name):
                raise ValueError(f"Feature named {name} does not exist in {namespace}")
            # Save data for this feature
            namespace, name = self.__class__._split_name(namespace, name)
            with conn.session_scope(self.session_maker) as session:
                feature = (
                    session.query(model.Feature)
                    .filter_by(name=name, namespace=namespace)
                    .one()
                )
                # Save individual feature
                feature.save(df)
        else:
            # Multiple features in column names
            for feature_name in feature_columns:
                if not self._exists(model.Feature, namespace, name):
                    raise ValueError(
                        f"Feature named {name} does not exist in {namespace}"
                    )
            for feature_name in feature_columns:
                # Save individual features
                feature_df = df[[*df.columns.difference(feature_columns), feature_name]]
                self.save_dataframe(feature_df)

    def last(self, features):
        """Fetch the last value of one or more features

        Args:
            features, str, list, pd.DataFrame: feature or features to fetch

        Returns:
            dict: of name, value pairs
        """
        result = {}
        for f in self._unpack_list(features):
            namespace, name = f
            with conn.session_scope(self.session_maker) as session:
                feature = (
                    session.query(model.Feature)
                    .filter_by(name=name, namespace=namespace)
                    .one_or_none()
                )
                if not feature:
                    raise ValueError(f"No feature named {name} exists in {namespace}")
                # Load individual feature
                result[f"{namespace}/{name}"] = feature.last(mode=self.mode)
        return result

    def create_task(self):
        """Create a scheduled task to update the feature store."""
        raise NotImplementedError()

    def update_task(self):
        """Update a task."""
        raise NotImplementedError()

    def delete_task(self):
        """Delete a task."""
        raise NotImplementedError()
