from abc import ABC
import pandas as pd


class BaseFeatureStore(ABC):
    """Base class for ByteHub feature stores."""

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
            return [cls._split_name(namespace=namespace, name=obj)]
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
                    r.append(cls._split_name(name=item, namespace=namespace))
                elif isinstance(item, dict):
                    r.append(
                        cls._split_name(
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

    def __init__(self, connection_string="sqlite:///bytehub.db", backend="pandas"):
        """Create a Feature Store, or connect to an existing one."""
        raise NotImplementedError()

    def list_namespaces(self, **kwargs):
        """List namespaces in the feature store.

        Search by name or regex query.

        Args:
            name (str, optional): name of namespace to filter by.
            namespace (str, optional): same as name.
            regex (str, optional): regex filter on name.

        Returns:
            pd.DataFrame: DataFrame of namespaces and metadata.
        """
        raise NotImplementedError()

    def create_namespace(self, name, **kwargs):
        """Create a new namespace in the feature store.

        Args:
            name (str): name of the namespace.
            description (str, optional): description for this namespace.
            url (str): url of data store.
            storage_options (dict, optional): storage options to be passed to Dask.
            meta (dict, optional): key/value pairs of metadata.
        """
        raise NotImplementedError()

    def update_namespace(self, name, **kwargs):
        """Update a namespace in the feature store.

        Args:
            name (str): namespace to update.
            description (str, optional): updated description.
            storage_options (dict, optional): updated storage options.
            meta (dict, optional): updated key/value pairs of metadata.
                To remove metadata, update using `{"key_to_remove": None}`.
        """
        raise NotImplementedError()

    def delete_namespace(self, name):
        """Delete a namespace from the feature store.

        Args:
            name: namespace to be deleted.
        """
        raise NotImplementedError()

    def clean_namespace(self, name):
        """Removes any data that is not associated with features in the namespace.
        Run this to free up disk space after deleting features

        Args:
            name (str): namespace to clean
        """
        raise NotImplementedError()

    def list_features(self, **kwargs):
        """List features in the feature store.

        Search by namespace, name and/or regex query

        Args:
            name (str, optional): name of feature to filter by.
            namespace (str, optional): namespace to filter by.
            regex (str, optional): regex filter on name.
            friendly (bool, optional): simplify output for user.

        Returns:
            pd.DataFrame: DataFrame of features and metadata.
        """
        raise NotImplementedError()

    def create_feature(self, name, namespace=None, **kwargs):
        """Create a new feature in the feature store.

        Args:
            name (str): name of the feature
            namespace (str, optional): namespace which should hold this feature.
            description (str, optional): description for this namespace.
            partition (str, optional): partitioning of stored timeseries (default: `"date"`).
            serialized (bool, optional): if `True`, converts values to JSON strings before saving,
                which can help in situations where the format/schema of the data changes
                over time.
            transform (str, optional): pickled function code for feature transforms.
            meta (dict, optional): key/value pairs of metadata.
        """
        raise NotImplementedError()

    def clone_feature(self, name, namespace=None, **kwargs):
        """Create a new feature by cloning an existing one.

        Args:
            name (str): name of the feature.
            namespace (str, optional): namespace which should hold this feature.
            from_name (str): the name of the existing feature to copy from.
            from_namespace (str, optional): namespace of the existing feature.
        """
        raise NotImplementedError()

    def delete_feature(self, name, namespace=None, delete_data=False):
        """Delete a feature from the feature store.

        Args:
            name (str): name of feature to delete.
            namespace (str, optional): namespace, if not included in feature name.
            delete_data (bool, optional): if set to `True` will delete underlying stored data
                for this feature, otherwise default behaviour is to delete the feature store
                metadata but leave the stored timeseries values intact.
        """
        raise NotImplementedError()

    def update_feature(self, name, namespace=None, **kwargs):
        """Update a namespace in the feature store.

        Args:
            name (str): feature to update.
            namespace (str, optional): namespace, if not included in feature name.
            description (str, optional): updated description.
            transform (str, optional): pickled function code for feature transforms.
            meta (dict, optional): updated key/value pairs of metadata.
                To remove metadata, update using `{"key_to_remove": None}`.
        """
        raise NotImplementedError()

    def transform(self, name, namespace=None, from_features=[]):
        """Decorator for creating/updating virtual (transformed) features.
        Use this on a function that accepts a dataframe input and returns an output dataframe
        of tranformed values.

        Args:
            name (str): feature to update.
            namespace (str, optional): namespace, if not included in feature name.
            from_features (list): list of features which should be transformed by this one
        """
        raise NotImplementedError()

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
            features (Union[str, list, pd.DataFrame]): name of feature to load, or list/DataFrame of feature namespaces/name.
            from_date (datetime, optional): start date to load timeseries from, defaults to everything.
            to_date (datetime, optional): end date to load timeseries to, defaults to everything.
            freq (str, optional): frequency interval at which feature values should be sampled.
            time_travel (str, optional): timedelta string, indicating that time-travel should be applied to the
                returned timeseries values, useful in forecasting applications.

        Returns:
            Union[pd.DataFrame, dask.DataFrame]: depending on which backend was specified in the feature store.
        """
        raise NotImplementedError()

    def save_dataframe(self, df, name=None, namespace=None):
        """Save a DataFrame of feature values to the feature store.

        Args:
            df (pd.DataFrame): DataFrame of feature values.
                Must have a `time` column or DateTimeIndex of time values.
                Optionally include a `created_time` column (defaults to `utcnow()` if omitted).
                For a single feature: a `value` column, or column header of feature `namespace/name`.
                For multiple features name the columns using `namespace/name`.
            name (str, optional): name of feature, if not included in DataFrame column name.
            namespace (str, optional): namespace, if not included in DataFrame column name.
        """
        raise NotImplementedError()

    def last(self, features):
        """Fetch the last value of one or more features.

        Args:
            features (Union[str, list, pd.DataFrame]): feature or features to fetch.

        Returns:
            dict: dictionary of name, last value pairs.
        """
        raise NotImplementedError()

    def create_task(self):
        """Create a scheduled task to update the feature store."""
        raise NotImplementedError()

    def update_task(self):
        """Update a task."""
        raise NotImplementedError()

    def delete_task(self):
        """Delete a task."""
        raise NotImplementedError()
