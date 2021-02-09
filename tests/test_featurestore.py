import bytehub as bh
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import random
import string
import posixpath
import pytest


def random_string(n):
    return "".join(random.choice(string.ascii_lowercase) for x in range(n))


class TestFeatureStore:
    def setup_class(self):
        print("Starting tests: creating feature store")
        # Creating feature store
        self.file_name = random_string(10)
        self.location = "/tmp"
        self.sqlite_file = "sqlite:////tmp/" + self.file_name + ".db"
        self.fs = bh.FeatureStore(self.sqlite_file)
        fs = self.fs
        fs.create_namespace("test", url=posixpath.join(self.location, self.file_name))

    def teardown_class(self):
        print("Finished tests: tearing down...")
        del self.fs
        try:
            os.remove(self.sqlite_file)
        except Exception:
            pass
        try:
            os.rmdir(self.location)
        except Exception:
            pass

    def test_utils(self):
        print("Testing utility functions")
        fs = self.fs

        assert fs._split_name(namespace="x", name="y") == ("x", "y")
        assert fs._split_name(namespace="x", name="y/z") == ("x", "y/z")
        assert fs._split_name(name="y/z") == ("y", "z")
        assert fs._split_name(name="z") == (None, "z")

        with pytest.raises(Exception):
            fs._validate_kwargs(
                dict.fromkeys(["x", "y"], 0), valid=["x"], mandatory=["x"]
            )
        with pytest.raises(Exception):
            fs._validate_kwargs(
                dict.fromkeys(["x", "y"], 0), valid=["x", "y", "z"], mandatory=["z"]
            )
        assert (
            fs._validate_kwargs(
                dict.fromkeys(["x", "z"], 0), valid=["x", "y", "z"], mandatory=["z"]
            )
            is None
        )
        assert (
            fs._validate_kwargs(dict.fromkeys(["x", "y"], 0), valid=["x", "y", "z"])
            is None
        )

        assert fs._unpack_list("test/test1") == [("test", "test1")]
        assert fs._unpack_list("test1", namespace="test") == [("test", "test1")]
        assert fs._unpack_list(["test1", "test2"], namespace="test") == [
            ("test", "test1"),
            ("test", "test2"),
        ]
        assert fs._unpack_list(["test/test1", "test/test2"]) == [
            ("test", "test1"),
            ("test", "test2"),
        ]
        assert fs._unpack_list(
            [{"name": "test/test1"}, {"name": "test2", "namespace": "test"}]
        ) == [("test", "test1"), ("test", "test2")]
        df = pd.DataFrame({"namespace": ["test", "test"], "name": ["test1", "test2"]})
        assert fs._unpack_list(df) == [("test", "test1"), ("test", "test2")]

    def test_namespaces(self):
        print("Testing namespaces...")
        fs = self.fs

        ns1 = random_string(5)
        ns2 = random_string(5)

        # Namespace without url should raise exception
        with pytest.raises(Exception):
            fs.create_namespace(ns1, description="ns1")

        fs.create_namespace(
            ns1, description="ns1", url=posixpath.join(self.location, ns1)
        )

        # Create with duplicate url should raise exception
        with pytest.raises(Exception):
            fs.create_namespace(
                ns2, description="ns2", url=posixpath.join(self.location, ns1)
            )

        fs.create_namespace(
            ns2, description="ns2", url=posixpath.join(self.location, ns2)
        )

        namespaces = fs.list_namespaces()
        assert ns1 in namespaces.name.tolist()
        assert ns2 in namespaces.name.tolist()
        assert "ns1" in namespaces.description.tolist()
        assert "ns2" in namespaces.description.tolist()

        # Update description
        fs.update_namespace(ns1, description="ns1-modified")
        namespaces = fs.list_namespaces()
        assert "ns1" not in namespaces.description.tolist()
        assert "ns1-modified" in namespaces.description.tolist()
        # Check version number got bumped
        assert namespaces.query("name == @ns1").version.iloc[0] == 2

        # Update non-existent feature
        with pytest.raises(Exception):
            fs.update_namespace("does-not-exist", description="ns1-modified")

        # Update metadata
        fs.update_namespace(ns1, meta={"key1": "value1"})
        fs.update_namespace(ns1, meta={"key2": "value2"})
        namespaces = fs.list_namespaces(name=ns1)
        assert len(namespaces) == 1
        assert "key1" in namespaces.meta[0].keys()
        assert "key2" in namespaces.meta[0].keys()
        # Remove key2 from metadata
        fs.update_namespace(ns1, meta={"key2": None})
        namespaces = fs.list_namespaces(name=ns1)
        assert "key1" in namespaces.meta[0].keys()
        assert "key2" not in namespaces.meta[0].keys()

        # Search namespaces
        namespaces = fs.list_namespaces(name=ns1)
        assert len(namespaces) == 1
        assert namespaces.name.iloc[0] == ns1
        namespaces = fs.list_namespaces(regex="test")
        assert namespaces.name.iloc[0] == "test"

        # Add feature to namespace
        fs.create_feature(f"{ns1}/test1")
        with pytest.raises(Exception):
            fs.delete_namespace(ns1)
        print(fs.list_features())
        fs.delete_feature(f"{ns1}/test1")

        fs.delete_namespace(ns1)
        fs.delete_namespace(ns2)
        namespaces = fs.list_namespaces()
        assert ns1 not in namespaces.name.tolist()
        assert ns2 not in namespaces.name.tolist()

    def test_features(self):
        print("Testing features...")
        fs = self.fs

        fs.create_feature("test/feature1", description="feature1")
        fs.create_feature("feature2", namespace="test", description="feature2")
        fs.create_namespace(
            "test2", description="test2", url=posixpath.join(self.location, "test2")
        )
        fs.create_feature("feature1", namespace="test2", description="feature2")

        # Duplicate feature
        with pytest.raises(Exception):
            fs.create_feature("test/feature1")
        with pytest.raises(Exception):
            fs.create_feature("feature1", namespace="test")

        features = fs.list_features(namespace="test")
        assert "feature1" in features.name.tolist()
        assert "feature2" in features.name.tolist()
        features = fs.list_features(namespace="test2")
        assert "feature1" in features.name.tolist()
        assert "feature2" not in features.name.tolist()
        features = fs.list_features(name="feature2")
        assert "test" in features.namespace.tolist()
        assert "test2" not in features.namespace.tolist()
        features = fs.list_features(regex="feature.")
        assert len(features) == 3

        fs.delete_feature("feature1", namespace="test")
        fs.delete_feature("feature2", namespace="test")
        with pytest.raises(Exception):
            fs.delete_feature("feature2", namespace="test")
        fs.delete_feature("feature1", namespace="test2")

        assert fs.list_features(namespace="test2").empty
        assert fs.list_features(namespace="test").empty

    def test_dataframes(self):
        print("Testing data load/save...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-03-01")
        df1 = pd.DataFrame({"time": dts, "value": np.random.randn(len(dts))}).set_index(
            "time"
        )
        dts = pd.date_range("2021-01-01", "2021-02-01", freq="60min")
        df2 = pd.DataFrame(
            {"time": dts, "value": [{"x": np.random.randn()} for x in dts]}
        )
        df3 = pd.DataFrame(
            {
                "time": dts,
                "test/df3": np.random.randn(len(dts)),
                "test/df4": [random_string(5) for x in dts],
            }
        )
        df5 = pd.DataFrame({"time": dts, "test/df5": np.random.randn(len(dts))})

        # Create features to hold these dataframes
        fs.create_feature("test/df1", description="df1")
        fs.create_feature("test/df2", description="df2")
        fs.create_feature("test/df3", description="df3")
        fs.create_feature("test/df4", description="df4", partition="year")

        # Save to non-existent feature
        with pytest.raises(Exception):
            fs.save_dataframe(df1, "test/df5")
        with pytest.raises(Exception):
            fs.save_dataframe(df5)

        fs.save_dataframe(df1, "test/df1")
        fs.save_dataframe(df2, "df2", namespace="test")
        fs.save_dataframe(df3)
        # Try re-writing df1
        fs.save_dataframe(df1, "test/df1")

        # Load back and check
        assert fs.load_dataframe("test/df1").equals(
            df1.rename(columns={"value": "test/df1"})
        )
        assert fs.load_dataframe("test/df2").equals(
            df2.set_index("time").rename(columns={"value": "test/df2"})
        )
        assert fs.load_dataframe(["test/df3", "test/df4"]).equals(df3.set_index("time"))

        # Delete features
        fs.delete_feature("test/df1")
        fs.delete_feature("test/df2")
        fs.delete_feature("test/df3")
        fs.delete_feature("test/df4")

    def test_resampling(self):
        print("Testing data resampling...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-03-01")
        df1 = pd.DataFrame(
            {"time": dts, "test/resample1": np.random.randn(len(dts))}
        ).set_index("time")
        dts = pd.date_range("2021-01-01", "2021-02-01", freq="60min")
        df2 = pd.DataFrame(
            {"time": dts, "test/resample2": [{"x": np.random.randn()} for x in dts]}
        ).set_index("time")

        fs.create_feature("test/resample1", description="df1")
        fs.create_feature("test/resample2", description="df2")
        fs.save_dataframe(df1)
        fs.save_dataframe(df2)

        result = fs.load_dataframe(["test/resample1", "test/resample2"])
        result_dask = fs.load_dataframe(
            ["test/resample1", "test/resample2"], mode="dask"
        )
        assert result.equals(pd.concat([df1, df2], join="outer", axis=1).ffill())
        assert result.equals(result_dask.compute())
        result = fs.load_dataframe(["test/resample1", "test/resample2"], freq="2d")
        result_dask = fs.load_dataframe(
            ["test/resample1", "test/resample2"], freq="2d", mode="dask"
        )
        assert result.equals(
            pd.concat([df1, df2], join="outer", axis=1).resample("2d").ffill().ffill()
        )
        assert result.equals(result_dask.compute())
        result = fs.load_dataframe(["test/resample1", "test/resample2"], freq="10min")
        result_dask = fs.load_dataframe(
            ["test/resample1", "test/resample2"], freq="10min", mode="dask"
        )
        assert result.equals(
            pd.concat([df1, df2], join="outer", axis=1)
            .resample("10min")
            .ffill()
            .ffill()
        )
        assert result.equals(result_dask.compute())
        result = fs.load_dataframe(
            ["test/resample1", "test/resample2"],
            freq="10min",
            from_date="2021-01-10",
            to_date="2021-01-12",
        )
        result_dask = fs.load_dataframe(
            ["test/resample1", "test/resample2"],
            freq="10min",
            from_date="2021-01-10",
            to_date="2021-01-12",
            mode="dask",
        )
        compare = (
            pd.concat([df1, df2], join="outer", axis=1)
            .resample("10min")
            .ffill()
            .ffill()
        )
        compare = compare[
            (compare.index >= pd.Timestamp("2021-01-10"))
            & (compare.index <= pd.Timestamp("2021-01-12"))
        ]
        assert result.equals(compare)
        assert result.equals(result_dask.compute())
        # Use dataframe to specify which features to load
        result = fs.load_dataframe(
            fs.list_features(regex=r"resample."),
            freq="10min",
            from_date="2021-01-10",
            to_date="2021-01-12",
        )
        assert result.equals(compare)
        result = fs.load_dataframe(
            "test/resample1",
            from_date="2021-01-10",
            to_date="2021-01-12",
        )
        result_dask = fs.load_dataframe(
            "test/resample1",
            from_date="2021-01-10",
            to_date="2021-01-12",
            mode="dask",
        )
        compare = df1[
            (df1.index >= pd.Timestamp("2021-01-10"))
            & (df1.index <= pd.Timestamp("2021-01-12"))
        ]
        assert result.equals(compare)
        assert result.equals(result_dask.compute())

        fs.delete_feature("test/resample1")
        fs.delete_feature("test/resample2")

    def test_empty_features(self):
        print("Testing empty feature datasets...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-03-01")
        df1 = pd.DataFrame(
            {"time": dts, "test/empty1": np.random.randn(len(dts))}
        ).set_index("time")
        fs.create_feature("test/empty1")

        result = fs.load_dataframe(["test/empty1"])
        assert result.empty
        result = fs.load_dataframe(
            ["test/empty1"], from_date="2021-01-01", to_date="2021-03-01", freq="1d"
        )
        assert len(result) == len(dts)

        fs.save_dataframe(df1)
        # Load data outside of time range
        result = fs.load_dataframe(
            ["test/empty1"], from_date="2020-01-01", to_date="2020-03-01"
        )
        assert result.empty

        fs.delete_feature("test/empty1")

    def test_time_travel(self):
        print("Testing time travel...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-03-01")
        df1 = pd.DataFrame(
            {
                "time": dts,
                "test/timetravel1": np.random.randint(0, 100, size=len(dts)),
                "created_time": dts - pd.Timedelta("10min"),
            }
        ).set_index("time")
        df2 = pd.DataFrame(
            {
                "time": dts,
                "test/timetravel1": np.random.randint(0, 100, size=len(dts)),
                "created_time": dts - pd.Timedelta("30min"),
            }
        ).set_index("time")
        df3 = pd.DataFrame(
            {
                "time": dts,
                "test/timetravel1": np.random.randint(0, 100, size=len(dts)),
                "created_time": dts - pd.Timedelta("60min"),
            }
        ).set_index("time")
        fs.create_feature("test/timetravel1")

        fs.save_dataframe(df1)
        fs.save_dataframe(df2)
        fs.save_dataframe(df3)

        result = fs.load_dataframe("test/timetravel1")
        assert result.equals(df1.drop(columns="created_time"))
        result = fs.load_dataframe("test/timetravel1", time_travel="-15min")
        assert result.equals(df2.drop(columns="created_time"))
        result = fs.load_dataframe("test/timetravel1", time_travel="-60min")
        assert result.equals(df3.drop(columns="created_time"))
        result = fs.load_dataframe("test/timetravel1", time_travel="-120min")
        assert result.empty

        fs.delete_feature("test/timetravel1")

    def test_last_values(self):
        print("Testing last feature values...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-03-01")
        df1 = pd.DataFrame(
            {
                "time": dts,
                "test/last1": np.random.randint(0, 100, size=len(dts)),
                "test/last2": np.random.randint(0, 100, size=len(dts)),
            }
        ).set_index("time")

        fs.create_feature("test/last1")
        fs.create_feature("test/last2")
        fs.create_feature("test/last3")

        fs.save_dataframe(df1)

        result = fs.last("test/last1")
        assert result == {"test/last1": df1["test/last1"].values[-1]}
        result = fs.last("test/last3")
        assert result == {"test/last3": None}
        result = fs.last(fs.list_features(regex=r"last."))
        assert result == {
            "test/last1": df1["test/last1"].values[-1],
            "test/last2": df1["test/last2"].values[-1],
            "test/last3": None,
        }

        fs.delete_feature("test/last1")
        fs.delete_feature("test/last2")
        fs.delete_feature("test/last3")
