import sqlalchemy as sa
from sqlalchemy import Table, Column, ForeignKey
from sqlalchemy import Integer, String, Boolean, JSON, Enum
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
import re
import copy
import types
from . import _utils as utils
from . import _timeseries as ts
from . import _connection as conn


Base = declarative_base()
valid_name = re.compile(r"^[a-zA-Z0-9\.#_-]+$")
partitions = Enum("year", "date", name="partition")


class FeatureStoreMixin(object):
    name = Column(String(128), primary_key=True, nullable=False)
    description = Column(String, default="")
    meta = Column(JSON, default={})
    version = Column(Integer, default=1, nullable=False)

    @validates("name")
    def _validate_name(self, key, value):
        if not valid_name.match(value):
            raise ValueError(f"Invalid name {value}")
        return value

    def as_dict(self):
        return {
            k: v if utils.is_jsonable(v) else str(v)
            for k, v in self.__dict__.items()
            if k[0] != "_"
        }

    def bump_version(self):
        if self.version:
            self.version += 1
        else:
            self.version = 1

    def update_from_dict(self, payload):
        if not payload:
            return
        if "name" in payload and self.name:
            raise ValueError(
                f"Cannot change name of {self.__class__.__name__}: use clone instead"
            )
        if "namespace" in payload and self.namespace:
            raise ValueError(
                f"Cannot change namespace of {self.__class__.__name__}: use clone instead"
            )
        for key, value in payload.items():
            if key == "meta" or key == "metadata":
                # See https://amercader.net/blog/beware-of-json-fields-in-sqlalchemy/
                self.meta = copy.deepcopy(self.meta)
                # Merge old metadata with new
                if self.meta:
                    self.meta.update(value)
                else:
                    self.meta = value
                # Remove any keys that no longer have values
                self.meta = {k: v for k, v in self.meta.items() if v is not None}
            else:
                # Update fields
                setattr(self, key, value) if hasattr(self, key) else None
        self.bump_version()


class Namespace(Base, FeatureStoreMixin):
    __tablename__ = "namespace"

    url = Column(String, nullable=False, unique=True)
    storage_options = Column(JSON, nullable=False, default={})

    @hybrid_property
    def namespace(self):
        return self.name

    @namespace.setter
    def namespace(self, value):
        self.name = value

    @namespace.expression
    def namespace(cls):
        return cls.name

    def clean(self):
        # Check for unused data and remove it
        feature_paths = ts.feature_paths(self.url, storage_options=self.storage_options)
        active_feature_names = [f.name for f in self.features]
        feature_data = [f.split("/")[-1] for f in feature_paths]
        for feature in feature_data:
            if feature not in active_feature_names:
                # Redundant data... delete it
                ts.delete(feature, self.url, storage_options=self.storage_options)


class Feature(Base, FeatureStoreMixin):
    __tablename__ = "feature"

    namespace = Column(String(128), ForeignKey("namespace.name"), primary_key=True)
    namespace_object = relationship("Namespace", backref="features")

    partition = Column(partitions, default="date", nullable=False)
    serialized = Column(Boolean, default=False, nullable=False)
    transform = Column(JSON, nullable=True)

    @hybrid_property
    def full_name(self):
        return f"{self.namespace}/{self.name}"

    @validates("serialized")
    def validate_serialized(self, key, value):
        if self.serialized is not None and value != self.serialized:
            raise ValueError("Cannot change serialized setting on existing feature")
        return value

    @validates("transform")
    def validate_transform(self, key, value):
        if not value:
            return value
        if isinstance(value.get("function"), str):
            # Function already serialized, no conversion required
            func = value.get("function")
        elif isinstance(value.get("function"), types.FunctionType):
            func = utils.serialize(value["function"])
        else:
            raise ValueError(
                "Transform must be a Python function, accepting a single dataframe input"
            )
        assert "function" in value.keys(), "Transform must have a function defined"
        assert "args" in value.keys(), "Transform must have arguments defined"
        # Convert function to base64/cloudpickle format
        return {
            "format": "cloudpickle",
            "function": func,
            "args": value["args"],
        }

    @classmethod
    def clone_from(cls, other, namespace, name):
        if not isinstance(other, cls):
            raise ValueError(f"Must clone from another {cls.__name__}")
        clone = cls()
        # Build new Feature with same settings as old
        clone.namespace = namespace
        clone.name = name
        payload = other.as_dict()
        payload.pop("namespace")
        payload.pop("name")
        payload.pop("version")
        clone.update_from_dict(payload)
        return clone

    def save(self, df):
        ts.save(
            df,
            self.name,
            self.namespace_object.url,
            self.namespace_object.storage_options,
            partition=self.partition,
            serialized=self.serialized,
        )

    def load_transform(
        self, from_date, to_date, freq, time_travel, mode, n_partitions=None, callers=[]
    ):
        # Get the SQLAlchemy session for this feature
        session = sa.inspect(self).session
        if not session:
            raise RuntimeError(f"{self.name} is not bound to an SQLAlchemy session")
        # Check for recursive transforms
        if self.full_name in callers:
            raise RuntimeError(
                f"Recursive feature transform detected on {self.full_name}"
            )
        # Load the transform function
        func = utils.deserialize(self.transform["function"])
        # Load the features to transform
        dfs = []
        # Load each requested feature
        for f in self.transform["args"]:
            namespace, name = f.split("/")[0], "/".join(f.split("/")[1:])
            feature = (
                session.query(Feature)
                .filter_by(name=name, namespace=namespace)
                .one_or_none()
            )
            if not feature:
                raise ValueError(f"No feature named {name} exists in {namespace}")
            # Load individual feature
            df = feature.load(
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                mode=mode,
                n_partitions=n_partitions,
                callers=[*callers, self.full_name],
            )
            dfs.append(df.rename(columns={"value": f"{namespace}/{name}"}))
        # Merge features into a single dataframe
        dfs = ts.concat(dfs)
        # Make sure columns are in the same order as args
        dfs = dfs[self.transform["args"]]
        # Apply transform function
        transformed = ts.transform(dfs, func, mode)
        return transformed

    def load(
        self,
        from_date=None,
        to_date=None,
        freq=None,
        time_travel=None,
        mode="pandas",
        n_partitions=None,
        callers=[],
    ):
        # Does this feature need to be transformed?
        if self.transform:
            return self.load_transform(
                from_date=from_date,
                to_date=to_date,
                freq=freq,
                time_travel=time_travel,
                mode=mode,
                n_partitions=n_partitions,
                callers=callers,
            )
        # Get location
        url = self.namespace_object.url
        storage_options = self.namespace_object.storage_options
        # Restrict which partitions are loaded (used for getting last partition)
        partitions = (
            ts.list_partitions(
                self.name, url, storage_options, n=n_partitions, reverse=True
            )
            if n_partitions
            else None
        )
        # Load dataframe
        return ts.load(
            self.name,
            url,
            storage_options,
            from_date,
            to_date,
            freq,
            time_travel,
            mode,
            self.serialized,
            partitions=partitions,
        )

    def last(self, mode="pandas"):
        # Fetch last feature value
        df = self.load(mode=mode)
        result = df.tail(1)
        if result.empty:
            return None
        else:
            return result["value"].iloc[0]

    def delete_data(self):
        # Deletes all of the data on this feature
        ts.delete(
            self.name, self.namespace_object.url, self.namespace_object.storage_options
        )

    def import_data_from(self, other):
        # Copy data over from another feature
        if not isinstance(other, self.__class__):
            raise ValueError(f"Must clone from another {cls.__name__}")
        # Get location of other feature to copy from
        url = other.namespace_object.url
        storage_options = other.namespace_object.storage_options
        # Copy data to new location
        ts.copy(
            other.name,
            url,
            storage_options,
            self.name,
            self.namespace_object.url,
            self.namespace_object.storage_options,
        )
