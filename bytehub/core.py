import sqlalchemy as sa
import pandas as pd
import posixpath
import json
from sqlalchemy.sql import text
from . import _connection as conn
from ._base import BaseFeatureStore
from . import _timeseries as ts
from .exceptions import *


try:
    # Allow for a minimal install with no dask
    from . import _model as model
except ImportError:
    pass


class CoreFeatureStore(BaseFeatureStore):
    """**Core Feature Store**

    Connects directly to a SQLAlchemy-compatible database.

    When using specifying features
    for `create_feature`, `update_feature`, etc., use either:

    * `namespace` and `name` as arguments; or
    * specify `name` in the format `"my-namespace/my-feature"`.
    """

    def __init__(
        self,
        connection_string="sqlite:///bytehub.db",
        backend="pandas",
        connect_args={},
    ):
        """
        Args:
            connection_string (str): SQLAlchemy connection string for database
                containing feature store metadata - defaults to local sqlite file.
            backend (str, optional): either `"pandas"` (default) or `"dask"`, specifying the type
                of dataframes returned by `load_dataframe`.
            connect_args (dict, optional): dictionary of [connection arguments](https://docs.sqlalchemy.org/en/14/core/engines.html#sqlalchemy.create_engine.params.connect_args)
                to pass to SQLAlchemy.
        """
        self.engine, self.session_maker = conn.connect(
            connection_string, connect_args=connect_args
        )
        if backend.lower() not in ["pandas", "dask"]:
            raise FeatureStoreException("Backend must be either pandas or dask")
        self.mode = backend.lower()
        model.Base.metadata.create_all(self.engine)

    def _list(self, cls, namespace=None, name=None, regex=None, friendly=True):
        namespace, name = self.__class__._split_name(namespace, name)
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
            if "transform" in df.columns and friendly:
                df = df.assign(transform=df["transform"].apply(lambda x: x is not None))
            # Sort the columns
            column_order = ["namespace", "name", "version", "description", "meta"]
            column_order = [c for c in column_order if c in df.columns]
            df = df[[*column_order, *df.columns.difference(column_order)]]
            return df

    def _delete(self, cls, namespace=None, name=None, delete_data=False):
        namespace, name = self.__class__._split_name(namespace, name)
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(
                    f"No existing {cls.__name__} named {name} in {namespace}"
                )
            if hasattr(obj, "delete_data") and delete_data:
                obj.delete_data()
            session.delete(obj)

    def _update(self, cls, namespace=None, name=None, payload={}):
        namespace, name = self.__class__._split_name(namespace, name)
        with conn.session_scope(self.session_maker) as session:
            r = session.query(cls)
            if namespace:
                r = r.filter_by(namespace=namespace)
            if name:
                r = r.filter_by(name=name)
            obj = r.one_or_none()
            if not obj:
                raise MissingFeatureException(
                    f"No existing {cls.__name__} named {name} in {namespace}"
                )
            # Apply updates from payload
            obj.update_from_dict(payload)

    def _create(self, cls, namespace=None, name=None, payload={}):
        if cls is model.Namespace:
            payload.update({"name": name})
        else:
            namespace, name = self.__class__._split_name(namespace, name)
            if not self._exists(model.Namespace, namespace=namespace):
                raise MissingFeatureException(f"{namespace} namespace does not exist")
            payload.update({"name": name, "namespace": namespace})
        with conn.session_scope(self.session_maker) as session:
            obj = cls()
            obj.update_from_dict(payload)
            session.add(obj)

    def _exists(self, cls, namespace=None, name=None):
        ls = self._list(cls, namespace=namespace, name=name)
        return not ls.empty

    def list_namespaces(self, **kwargs):
        self.__class__._validate_kwargs(kwargs, ["name", "namespace", "regex"])
        return self._list(
            model.Namespace,
            name=kwargs.get("name", kwargs.get("namespace")),
            regex=kwargs.get("regex"),
        )

    def create_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "url", "storage_options", "meta"],
            mandatory=["url"],
        )
        self._create(model.Namespace, name=name, payload=kwargs)

    def update_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "storage_options", "meta"],
        )
        self._update(model.Namespace, name=name, payload=kwargs)

    def delete_namespace(self, name):
        if not self.list_features(namespace=name).empty:
            raise FeatureStoreException(
                f"{name} still contains features: these must be deleted first"
            )
        self._delete(model.Namespace, name=name)

    def clean_namespace(self, name):
        with conn.session_scope(self.session_maker) as session:
            r = session.query(model.Namespace)
            r = r.filter_by(name=name)
            namespace = r.one_or_none()
            if not namespace:
                raise MissingFeatureException(f"No existing Namespace named {name}")
            namespace.clean()

    def list_features(self, **kwargs):
        self.__class__._validate_kwargs(
            kwargs, valid=["name", "namespace", "regex", "friendly"]
        )
        return self._list(
            model.Feature,
            namespace=kwargs.get("namespace"),
            name=kwargs.get("name"),
            regex=kwargs.get("regex"),
            friendly=kwargs.get("friendly", True),
        )

    def create_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "partition", "serialized", "transform"],
            mandatory=[],
        )
        self._create(model.Feature, namespace=namespace, name=name, payload=kwargs)

    def clone_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["from_namespace", "from_name"],
            mandatory=["from_name"],
        )
        from_namespace, from_name = self.__class__._split_name(
            kwargs.get("from_namespace"), kwargs.get("from_name")
        )
        to_namespace, to_name = self.__class__._split_name(namespace, name)
        if not self._exists(model.Namespace, namespace=from_namespace):
            raise MissingFeatureException(f"{from_namespace} namespace does not exist")
        if not self._exists(model.Namespace, namespace=to_namespace):
            raise MissingFeatureException(f"{to_namespace} namespace does not exist")
        with conn.session_scope(self.session_maker) as session:
            # Get the existing feature
            r = session.query(model.Feature)
            r = r.filter_by(namespace=from_namespace, name=from_name)
            feature = r.one_or_none()
            if not feature:
                raise MissingFeatureException(
                    f"No existing Feature named {from_name} in {from_namespace}"
                )
            # Create the new feature
            new_feature = model.Feature.clone_from(feature, to_namespace, to_name)
            session.add(new_feature)
            # Copy data to new feature, if this raises exception will rollback
            if not new_feature.transform:
                new_feature.import_data_from(feature)

    def delete_feature(self, name, namespace=None, delete_data=False):
        self._delete(model.Feature, namespace, name, delete_data=delete_data)

    def update_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "transform"],
        )
        self._update(model.Feature, name=name, namespace=namespace, payload=kwargs)

    def transform(self, name, namespace=None, from_features=[]):
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
                    raise MissingFeatureException(
                        f"No feature named {name} exists in {namespace}"
                    )
                # Load individual feature
                df = feature.load(
                    from_date=from_date,
                    to_date=to_date,
                    freq=freq,
                    time_travel=time_travel,
                    mode=self.mode,
                )
                dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        return ts.concat(dfs)

    def save_dataframe(self, df, name=None, namespace=None):
        # Check dataframe columns
        feature_columns = df.columns.difference(["time", "created_time"])
        if len(feature_columns) == 1:
            # Single feature to save
            if feature_columns[0] == "value":
                if not name:
                    raise FeatureStoreException("Must specify feature name")
            else:
                name = feature_columns[0]
                df = df.rename(columns={name: "value"})
            if not self._exists(model.Feature, namespace, name):
                raise MissingFeatureException(
                    f"Feature named {name} does not exist in {namespace}"
                )
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
                    raise MissingFeatureException(
                        f"Feature named {name} does not exist in {namespace}"
                    )
            for feature_name in feature_columns:
                # Save individual features
                feature_df = df[[*df.columns.difference(feature_columns), feature_name]]
                self.save_dataframe(feature_df)

    def last(self, features):
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
                    raise MissingFeatureException(
                        f"No feature named {name} exists in {namespace}"
                    )
                # Load individual feature
                result[f"{namespace}/{name}"] = feature.last(mode=self.mode)
        return result
