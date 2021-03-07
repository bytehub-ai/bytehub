"""
ByteHub provides an easy-to-use Feature Store, optimised for time-series data.
It requires no complex infrastructure setup, and can be run locally, connected to
a remote database/file storage, or run in a cloud-hosted mode.
ByteHub uses Dask for data storage, allowing it to be scaled to large datasets and
cluster compute environments.

Exammple usage for a local SQLite feature store:

    import bytehub as bh
    fs = bh.FeatureStore()

Remote feature stores can be accessed using a SQLAlchemy connection string, e.g.:

    fs = bh.FeatureStore('postgresql+psycopg2://user:pass@host:port/bytehub')

Cloud-hosted feature stores can be accessed via a REST API endpoint:

    fs = bh.FeatureStore('https://api.bytehub.ai/')

or simply use the following to access ByteHub's cloud service:

    fs = bh.CloudFeatureStore()

See [https://docs.bytehub.ai](https://docs.bytehub.ai) for examples and tutorials.

"""
from .core import CoreFeatureStore
from .cloud import CloudFeatureStore
from ._version import __version__


def FeatureStore(connection_string="sqlite:///bytehub.db", backend="pandas", **kwargs):
    """Factory method to create Feature Store objects.

    Args:
        connection_string (str): SQLAlchemy connection string for database
            containing feature store metadata (defaults to local sqlite file)
            or an HTTPS endpoint to a cloud-hosted feature store.
        backend (str): either `"pandas"` (default) or `"dask"`, specifying the type
                of dataframes returned by `load_dataframe`.
        **kwargs: Additional options to be passed to the Feature Store constructor.

    Returns:
        Union[CoreFeatureStore, CloudFeatureStore]: Feature Store object.
    """
    if connection_string.startswith("http"):
        # Connect to cloud-hosted feature store
        return CloudFeatureStore(
            connection_string=connection_string, backend=backend, **kwargs
        )
    else:
        # Direct connection to database using CoreFeatureStore
        return CoreFeatureStore(
            connection_string=connection_string, backend=backend, **kwargs
        )
