from abc import ABC, abstractmethod
import functools
import types


class BaseStore(ABC):
    """Base storage class for ByteHub timeseries data.
    Sub-class this to add additional storage backends.
    """

    def __init__(self, url, storage_options={}):
        self.url = url
        self.storage_options = storage_options

    @abstractmethod
    def ls(self):
        """List features contained in storage."""
        raise NotADirectoryError()

    @abstractmethod
    def load(
        self, name, from_date=None, to_date=None, freq=None, time_travel=None, **kwargs
    ):
        """Load a single timeseries dataframe from storage."""
        raise NotImplementedError()

    @abstractmethod
    def save(self, name, df, **kwargs):
        """Save a timeseries dataframe."""
        raise NotImplementedError()

    @abstractmethod
    def first(self, name, **kwargs):
        """Retrieves first index from the timeseries."""
        raise NotImplementedError()

    def last(self, name, **kwargs):
        """Retrieves last index from the timeseries."""
        raise NotImplementedError()

    @abstractmethod
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
        if isinstance(ddf, types.GeneratorType):
            for ddf_ in ddf:
                destination_store._import(to_name, ddf_)
        else:
            destination_store._import(to_name, ddf)

    @abstractmethod
    def _export(self, name):
        """Export a timeseries as standardised dataframe."""
        raise NotImplementedError()

    @abstractmethod
    def _import(self, name, ddf):
        """Import a timeseries from standardised dataframe."""
        raise NotImplementedError()
