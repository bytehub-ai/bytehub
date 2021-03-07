"""
Exception raised by Feature Store objects
"""


class FeatureStoreException(Exception):
    """General feature store exception."""

    pass


class RemoteFeatureStoreException(Exception):
    """General feature store exception."""

    pass


class MissingFeatureException(Exception):
    """The requested feature/namespace does not exist."""

    pass
