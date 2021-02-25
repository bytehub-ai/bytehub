from .core import CoreFeatureStore
from .cloud import CloudFeatureStore
from ._version import __version__


def FeatureStore(connection_string="sqlite:///bytehub.db", backend="pandas", **kwargs):
    """Factory method to create Feature Store objects.

    Args:
        connection_string, str: SQLAlchemy connection string for database
            containing feature store metadata - defaults to local sqlite file
            or an HTTPS endpoint to a cloud-hosted feature store
        backend, str: eith 'pandas' (default) or 'dask', specifying the type
            of dataframes returned by load_dataframe
        **kwargs: Additional options to be passed to the Feature Store constructor

    Returns:
        CoreFeatureStore or CloudFeatureStore object
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
