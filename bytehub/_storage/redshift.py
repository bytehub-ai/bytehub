from ._base import BaseStore
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import awswrangler as wr
import redshift_connector


class Store(BaseStore):
    """AWS Redshift storage backend"""

    def __init__(self, url, storage_options={}):
        super().__init__(url, storage_options=storage_options)
        if "/" in url:
            host = url.split("/")[0]
            database = url.split("/")[1]
        else:
            host = url
            database = storage_options.get("database")
        # Connect to Redshift
        self.conn = redshift_connector.connect(
            host=host, database=database, **storage_options
        )

    def ls(self):
        raise NotImplementedError()

    def load(
        self, name, from_date=None, to_date=None, freq=None, time_travel=None, **kwargs
    ):
        raise NotImplementedError()

    def save(self, name, df, **kwargs):
        raise NotImplementedError()

    def first(self, name, **kwargs):
        raise NotImplementedError()

    def last(self, name, **kwargs):
        raise NotImplementedError()

    def delete(self, name):
        raise NotImplementedError()

    def _export(self, name):
        raise NotImplementedError()

    def _import(self, name, ddf):
        raise NotImplementedError()
