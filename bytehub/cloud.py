import pandas as pd
import posixpath
from getpass import getpass
import time
import os
import json
from ._base import BaseFeatureStore
from . import _timeseries as ts

try:
    # Allow for a minimal install with no requests
    import requests
    from requests_oauthlib import OAuth2Session
except ImportError:
    pass


class CloudFeatureStore(BaseFeatureStore):
    """Cloud Feature Store
    Connects to a hosted feature store via REST API.
    """

    def __init__(
        self,
        connection_string="https://api.bytehub.ai",
        backend="pandas",
        enable_transforms=False,
    ):
        """Connect to a cloud-hosted feature store

        Args:
            connection_string, str: URL of ByteHub Cloud, e.g. https://api.bytehub.ai
            backend, str: either 'pandas' (default) or 'dask', specifying the type
                of dataframes returned by load_dataframe
            enable_transforms, bool: whether to allow execution of pickled functions
                stored in the feature store - only enable if you trust the store
        """
        if backend.lower() not in ["pandas", "dask"]:
            raise ValueError("Backend must be either pandas or dask")
        self.mode = backend.lower()
        self._endpoint = connection_string
        if "/v1" not in self._endpoint:
            # Add API version
            self._endpoint = posixpath.join(self._endpoint, "v1/")
        if self._endpoint[-1] != "/":
            self._endpoint += "/"
        # Get the Oauth2 URLs
        response = requests.get(self._endpoint)
        self._check_response(response)
        self._urls = response.json()
        self._client_id = self._urls.pop("client_id")
        # Decide how to authenticate
        if os.environ.get("BYTEHUB_TOKEN"):
            # Non-interactive login using refresh token
            oauth = OAuth2Session(
                self.client_id, token={"refresh_token": os.environ.get("BYTEHUB_TOKEN")}
            )
            tokens = oauth.refresh_token(
                self._urls["token_url"],
                client_id=self._client_id,
                client_secret=None,
                include_client_id=True,
            )
        else:
            # Use interactive login
            oauth = OAuth2Session(
                self._client_id, redirect_uri=self._urls["callback_url"]
            )
            authorization_url, state = oauth.authorization_url(self._urls["login_url"])
            print(
                f"Please go to {authorization_url} and login. Copy the response code and paste below."
            )
            code_response = getpass("Response: ")
            tokens = oauth.fetch_token(
                self._urls["token_url"], code=code_response, include_client_id=True
            )
        self._tokens = tokens

    def _check_response(self, response):
        # Raise an exception if response is not OK
        try:
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            try:
                payload = response.json()
                message = payload.get("message", payload.get("Message", ""))
            except requests.exceptions.RequestException:
                message = response.text
            raise RuntimeError(message)

    def _check_tokens(self):
        # Check that token hasn't expired
        if time.time() < self._tokens["expires_at"] - 10:
            return True
        else:
            # Token expired... refresh it
            if "refresh_token" in self._tokens:
                oauth = OAuth2Session(self._client_id, token=self._tokens)
                tokens = oauth.refresh_token(
                    self._urls["token_url"],
                    client_id=self._client_id,
                    client_secret=None,
                    include_client_id=True,
                )
                self._tokens = tokens
            else:
                raise RuntimeError("Cannot find refresh token")

    def _api_headers(self):
        # Headers to add to API requests
        self._check_tokens()
        return {
            "Authorization": self._tokens["access_token"],
        }

    def _refresh(self, name=None):
        # Check access token expiry, and refresh if required
        df = self._namespaces
        if name:
            df = df[df.name == name]
        for idx, row in df.iterrows():
            # Get storage options
            opt = row["storage_options"]
            if "_expires" in opt and pd.Timestamp(
                opt["_expires"]
            ) < pd.Timestamp.utcnow() + pd.Timedelta("1min"):
                self.list_namespaces()
                return

    def _exists(self, entity, **kwargs):
        if entity.lower() == "feature":
            ls = self.list_features(
                namespace=kwargs.get("namespace"), name=kwargs.get("name")
            )
        elif entity.lower() == "namespace":
            ls = self.list_namespaces(name=kwargs.get("name", kwargs.get("namespace")))
        else:
            raise ValueError(f"Unrecognised entity: {entity}")
        return not ls.empty

    def _get(self, entity, **kwargs):
        if entity.lower() == "feature":
            ls = self.list_features(
                namespace=kwargs.get("namespace"), name=kwargs.get("name")
            )
        elif entity.lower() == "namespace":
            self._refresh(name=kwargs.get("name", kwargs.get("namespace")))
            ls = self._namespaces[
                self._namespaces.name == kwargs.get("name", kwargs.get("namespace"))
            ]
        else:
            raise ValueError(f"Unrecognised entity: {entity}")
        if len(ls) != 1:
            raise ValueError(f"{entity} not found: {kwargs}")
        return ls.iloc[0].to_dict()

    def list_namespaces(self, **kwargs):
        self.__class__._validate_kwargs(kwargs, ["name", "namespace", "regex"])
        url = self._endpoint + "namespace"
        response = requests.get(url, params=kwargs, headers=self._api_headers())
        self._check_response(response)
        df = pd.DataFrame(response.json())
        # Cache the namespace for use when loading/saving dataframes
        self._namespaces = df
        return df

    def create_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "url", "storage_options", "meta"],
            mandatory=["url"],
        )
        url = self._endpoint + "namespace"
        body = {"name": name, **kwargs}
        response = requests.post(url, json=body, headers=self._api_headers())
        self._check_response(response)
        # Call list_namespaces to refresh namespace cache
        self.list_namespaces()

    def update_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "storage_options", "meta"],
        )
        url = self._endpoint + "namespace"
        body = {"name": name, **kwargs}
        response = requests.patch(url, json=body, headers=self._api_headers())
        self._check_response(response)
        # Call list_namespaces to refresh namespace cache
        self.list_namespaces()

    def delete_namespace(self, name):
        if not self.list_features(namespace=name).empty:
            raise RuntimeError(
                f"{name} still contains features: these must be deleted first"
            )
        url = self._endpoint + "namespace"
        response = requests.delete(
            url, json={"name": name}, headers=self._api_headers()
        )
        self._check_response(response)
        # Call list_namespaces to refresh namespace cache
        self.list_namespaces()

    def clean_namespace(self, name):
        # Get namespace
        self._refresh(name=name)
        ns = self._get("namespace", name=name)
        # Check for unused data and remove it
        feature_paths = ts.feature_paths(ns["url"], ns["storage_options"])
        active_feature_names = self.list_features(namespace=name)
        active_feature_names = active_feature_names.name.tolist()
        feature_data = [f.split("/")[-1] for f in feature_paths]
        for feature in feature_data:
            if feature not in active_feature_names:
                # Redundant data... delete it
                ts.delete(feature, ns["url"], storage_options=ns["storage_options"])

    def list_features(self, **kwargs):
        self.__class__._validate_kwargs(kwargs, valid=["name", "namespace", "regex"])
        url = self._endpoint + "feature"
        response = requests.get(url, params=kwargs, headers=self._api_headers())
        self._check_response(response)
        df = pd.DataFrame(response.json())
        return df

    def create_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "partition", "serialized", "transform"],
            mandatory=[],
        )
        url = self._endpoint + "feature"
        body = {"name": name, "namespace": namespace, **kwargs}
        response = requests.post(url, json=body, headers=self._api_headers())
        self._check_response(response)

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
        if not self._exists("namespace", namespace=from_namespace):
            raise ValueError(f"{from_namespace} namespace does not exist")
        if not self._exists("namespace", namespace=to_namespace):
            raise ValueError(f"{to_namespace} namespace does not exist")
        # Get the existing feature
        payload = self._get("feature", namespace=from_namespace, name=from_name)
        _ = payload.pop("name")
        _ = payload.pop("namespace")
        # Create the new feature
        self.create_feature(
            name=to_name,
            namespace=to_namespace,
            **payload,
        )
        # Copy data to new feature, if this raises exception will rollback
        if not payload.transform:
            # Get location of this feature to copy to
            from_ns = self._get("namespace", name=from_namespace)
            to_ns = self._get("namespace", name=to_namespace)
            try:
                ts.copy(
                    from_name,
                    from_ns["url"],
                    from_ns["storage_options"],
                    to_name,
                    to_ns["url"],
                    to_ns["storage_options"],
                )
            except Exception as e:
                # Rollback... delete the newly created feature
                self.delete_feature(name=from_name, namespace=from_namespace)
                raise RuntimeError(f"Unable to save data to {path}: {str(e)}")

    def delete_feature(self, name, namespace=None, delete_data=False):
        url = self._endpoint + "feature"
        response = requests.delete(
            url,
            json={"name": name, "namespace": namespace},
            headers=self._api_headers(),
        )
        self._check_response(response)
        # Delete data
        if delete_data:
            namespace, name = self.__class__._split_name(namespace, name)
            ns = self._get("namespace", name=namespace)
            ts.delete(name, ns["url"], storage_options=ns["storage_options"])

    def update_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "transform"],
        )
        url = self._endpoint + "feature"
        body = {"name": name, "namespace": namespace, **kwargs}
        response = requests.patch(url, json=body, headers=self._api_headers())
        self._check_response(response)

    def transform(self, name, namespace=None, from_features=[]):
        def decorator(func):
            # Create or update feature with transform
            to_namespace, to_name = self._split_name(namespace=namespace, name=name)
            computed_from = [f"{ns}/{n}" for ns, n in self._unpack_list(from_features)]
            existing_features = self.list_features()
            existing_features = existing_features.apply(
                lambda x: x["namespace"] + "/" + x["name"], axis=1
            ).tolist()
            for feature in computed_from:
                assert (
                    feature in existing_features
                ), f"{feature} does not exist in the feature store"

            transform = {"function": func, "args": computed_from}
            payload = {"transform": transform, "description": func.__doc__}
            if self._exists("feature", namespace=to_namespace, name=to_name):
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

    def _load_transform(
        self, feature, from_date, to_date, freq, time_travel, callers=[]
    ):
        # Check for recursive transforms
        full_name = feature["namespace"] + "/" + feature["name"]
        if full_name in callers:
            raise RuntimeError(f"Recursive feature transform detected on {full_name}")
        # Load the transform function
        func = utils.deserialize(feature["transform"]["function"])
        # Load the features to transform
        dfs = []
        # Load each requested feature
        for f in feature["transform"]["args"]:
            namespace, name = f.split("/")[0], "/".join(f.split("/")[1:])
            f = self._get("feature", namespace=namespace, name=name)
            ns = self._get("namespace", name=namespace)
            # Load individual feature
            df = self._load(
                f, from_date, to_date, freq, time_travel, callers=[*callers, full_name]
            )
            dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        # Merge features into a single dataframe
        dfs = ts.concat(dfs)
        # Make sure columns are in the same order as args
        dfs = dfs[self.transform["args"]]
        # Apply transform function
        transformed = ts.transform(dfs, func, mode)
        return transformed

    def _load(self, feature, from_date, to_date, freq, time_travel, callers=[]):
        if feature["transform"]:
            # Apply feature transform
            return self._load_transform(
                feature, from_date, to_date, freq, time_travel, callers=[]
            )
        else:
            ns = self._get("namespace", name=feature["namespace"])
            return ts.load(
                feature["name"],
                ns["url"],
                ns["storage_options"],
                from_date,
                to_date,
                freq,
                time_travel,
                self.mode,
                feature["serialized"],
            )

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
            feature = self._get("feature", name=name, namespace=namespace)
            df = self._load(feature, from_date, to_date, freq, time_travel)
            dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        return ts.concat(dfs)

    def save_dataframe(self, df, name=None, namespace=None):
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
            feature = self._get("feature", namespace=namespace, name=name)
            # Save data for this feature
            namespace, name = self.__class__._split_name(namespace, name)
            ns = self._get("namespace", name=namespace)
            # Save individual feature
            ts.save(
                df,
                name,
                ns["url"],
                ns["storage_options"],
                partition=feature["partition"],
                serialized=feature["serialized"],
            )
        else:
            # Multiple features in column names
            for feature_name in feature_columns:
                namespace, name = self.__class__._split_name(namespace, feature_name)
                if not self._exists("feature", namespace=namespace, name=name):
                    raise ValueError(
                        f"Feature named {name} does not exist in {namespace}"
                    )
            for feature_name in feature_columns:
                # Save individual features
                feature_df = df[[*df.columns.difference(feature_columns), feature_name]]
                self.save_dataframe(feature_df)

    def last(self, features):
        pass