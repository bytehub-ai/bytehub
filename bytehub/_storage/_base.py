from abc import ABC
import functools
import pandas as pd
import dask.dataframe as dd


class BaseStore(ABC):
    """Base storage class for ByteHub timeseries data.
    Sub-class this to add additional storage backends.
    """

    def __init__(self, url, storage_options={}):
        self.url = url
        self.storage_options = storage_options

    def ls(self):
        """List features contained in storage."""
        raise NotADirectoryError()

    def load(
        self, name, from_date=None, to_date=None, freq=None, time_travel=None, **kwargs
    ):
        """Load a single timeseries dataframe from storage."""
        raise NotImplementedError()

    def save(self, name, df, **kwargs):
        """Save a timeseries dataframe."""
        raise NotImplementedError()

    def last(self, name, **kwargs):
        """Retrieves last index from the timeseries."""
        raise NotImplementedError()

    def delete(self, name):
        """Delete the data for a feature."""
        raise NotImplementedError()

    def copy(self, from_name, to_name, destination_store):
        """Used during clone operations to copy timeseries data between locations.
        Override this to implement more efficient copying between specific storage backends.
        """
        # Export existing data
        ddf = self._export(from_name)
        # Import to destination
        destination_store._import(to_name, ddf)

    def _export(self, name):
        """Export a timeseries as standardised dask dataframe."""
        raise NotImplementedError()

    def _import(self, name, ddf):
        """Import a timeseries from standardised dask dataframe."""
        raise NotImplementedError()
