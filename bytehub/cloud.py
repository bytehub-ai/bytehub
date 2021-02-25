import pandas as pd
from dask import dataframe as dd
import posixpath
import functools
import json
from ._base import BaseFeatureStore


class CloudFeatureStore(BaseFeatureStore):
    """Cloud Feature Store
    Connects directly to a hosted feature store via REST API.
    """

    def __init__(self, connection_string="sqlite:///bytehub.db", backend="pandas"):
        """Create a Feature Store, or connect to an existing one

        Args:
            connection_string, str: SQLAlchemy connection string for database
                containing feature store metadata - defaults to local sqlite file
            backend, str: eith 'pandas' (default) or 'dask', specifying the type
                of dataframes returned by load_dataframe
        """
        pass

    def list_namespaces(self, **kwargs):
        self.__class__._validate_kwargs(kwargs, ["name", "namespace", "regex"])
        pass

    def create_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "url", "storage_options", "meta"],
            mandatory=["url"],
        )
        pass

    def update_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "storage_options", "meta"],
        )
        pass

    def delete_namespace(self, name):
        if not self.list_features(namespace=name).empty:
            raise RuntimeError(
                f"{name} still contains features: these must be deleted first"
            )
        pass

    def clean_namespace(self, name):
        pass

    def list_features(self, **kwargs):
        self.__class__._validate_kwargs(kwargs, valid=["name", "namespace", "regex"])
        pass

    def create_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "partition", "serialized", "transform"],
            mandatory=[],
        )
        pass

    def clone_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["from_namespace", "from_name"],
            mandatory=["from_name"],
        )
        pass

    def delete_feature(self, name, namespace=None, delete_data=False):
        pass

    def update_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "transform"],
        )
        pass

    def transform(self, name, namespace=None, from_features=[]):
        pass

    def load_dataframe(
        self,
        features,
        from_date=None,
        to_date=None,
        freq=None,
        time_travel=None,
    ):
        pass

    def save_dataframe(self, df, name=None, namespace=None):
        pass

    def last(self, features):
        pass
