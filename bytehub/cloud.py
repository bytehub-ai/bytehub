import pandas as pd
from dask import dataframe as dd
import posixpath
import requests
import getpass
from requests_oauthlib import OAuth2Session
import functools
import json
from ._base import BaseFeatureStore


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
        """Create a Feature Store, or connect to an existing one

        Args:
            connection_string, str: URL of ByteHub Cloud, e.g. https://api.bytehub.ai
            backend, str: either 'pandas' (default) or 'dask', specifying the type
                of dataframes returned by load_dataframe
            enable_transforms, bool: whether to allow execution of pickled functions
                stored in the feature store - only enable if you trust the store
        """
        self._endpoint = connection_string
        if "/v1" not in self._endpoint:
            # Add API version
            self._endpoint = posixpath.join(self._endpoint, "v1/")
        if self._endpoint[-1] != "/":
            self._endpoint += "/"
        # Get the Oauth2 URLs
        response = requests.get(self.endpoint)
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
                self.urls["token_url"],
                client_id=self._client_id,
                client_secret=None,
                include_client_id=True,
            )
        else:
            # Use interactive login
            oauth = OAuth2Session(client_id, redirect_uri=self.urls["callback_url"])
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
            if "expires" in opt and pd.Timestamp(
                opt["expires"]
            ) < pd.Timestamp.utcnow() + pd.Timedelta("1min"):
                self.list_namespaces()
                return

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
        params = {"name": name, **kwargs}
        response = requests.post(url, params=params, headers=self._api_headers())
        self._check_response(response)
        # Call list_namespaces to refresh namespace cache
        self.list_namespaces()

    def update_namespace(self, name, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "storage_options", "meta"],
        )
        url = self._endpoint + "namespace"
        params = {"name": name, **kwargs}
        response = requests.patch(url, params=params, headers=self._api_headers())
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
            url, params={"name": name}, headers=self._api_headers()
        )
        self._check_response(response)
        # Call list_namespaces to refresh namespace cache
        self.list_namespaces()

    def clean_namespace(self, name):
        # TODO: Clean namespace
        raise NotImplementedError()

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
        params = {"name": name, "namespace": namespace, **kwargs}
        response = requests.post(url, params=params, headers=self._api_headers())
        self._check_response(response)

    def clone_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["from_namespace", "from_name"],
            mandatory=["from_name"],
        )
        raise NotImplementedError()

    def delete_feature(self, name, namespace=None, delete_data=False):
        url = self._endpoint + "feature"
        response = requests.delete(
            url,
            params={"name": name, "namespace": namespace},
            headers=self._api_headers(),
        )
        self._check_response(response)
        # TODO: Delete data

    def update_feature(self, name, namespace=None, **kwargs):
        self.__class__._validate_kwargs(
            kwargs,
            valid=["description", "meta", "transform"],
        )
        url = self._endpoint + "feature"
        params = {"name": name, "namespace": namespace, **kwargs}
        response = requests.patch(url, params=params, headers=self._api_headers())
        self._check_response(response)

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
