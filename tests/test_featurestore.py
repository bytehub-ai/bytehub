import bytehub as bh
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import random
import string
import fsspec
import posixpath
import json
import pytest


backends = [
    json.loads(os.environ[key])
    for key in os.environ.keys()
    if key.startswith("CLOUDSTORE_")
]
backends.append({"path": "/tmp", "storage_options": {}, "backend": "pandas"})
backends.append({"path": "/tmp", "storage_options": {}, "backend": "dask"})


def random_string(n):
    return "".join(random.choice(string.ascii_lowercase) for x in range(n))


def build_url(filesystem):
    fs = filesystem["fs"]
    location = filesystem["location"]
    if isinstance(fs.protocol, tuple):
        protocol = fs.protocol[0]
    else:
        protocol = fs.protocol
    return f"{protocol}://{location}"


def compare_df(df1, df2):
    if isinstance(df1, dd.DataFrame):
        df1 = df1.compute()
    if isinstance(df1, dd.DataFrame):
        df2 = df2.compute()
    return df1.equals(df2)


def compare_series(s1, s2):
    if isinstance(s1, dd.Series):
        s1 = s1.compute()
    if isinstance(s2, dd.Series):
        s2 = s2.compute()
    return s1.equals(s2)


def empty_df(df):
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    return df.empty


@pytest.fixture
def file_name():
    return random_string(10)


@pytest.fixture(params=backends)
def filesystem(request):
    path = request.param["path"]
    storage_options = request.param["storage_options"]
    storage_backend = request.param.get("backend", "pandas")

    fs, _, paths = fsspec.get_fs_token_paths(path, storage_options=storage_options)
    return {"fs": fs, "location": paths[0], "backend": storage_backend}


@pytest.fixture
def fs(filesystem, file_name):
    print(f"Starting tests using {filesystem['fs'].__class__.__name__}")
    # Creating feature store
    location = "/tmp"
    sqlite_file = "sqlite:////tmp/" + file_name + ".db"
    fs = bh.FeatureStore(sqlite_file)
    fs.create_namespace(
        "test",
        url=posixpath.join(build_url(filesystem), file_name),
        storage_options=filesystem["fs"].storage_options,
        backend=filesystem["backend"],
    )

    yield fs

    print("Finished tests: tearing down...")
    del fs
    try:
        os.remove(sqlite_file)
    except Exception:
        pass
    try:
        filesystem["fs"].rm(
            posixpath.join(filesystem["location"], file_name), recursive=True
        )
    except Exception:
        pass


def test_utils(fs):
    print("Testing utility functions")

    assert fs._split_name(namespace="x", name="y") == ("x", "y")
    assert fs._split_name(namespace="x", name="y/z") == ("x", "y/z")
    assert fs._split_name(name="y/z") == ("y", "z")
    assert fs._split_name(name="z") == (None, "z")

    with pytest.raises(Exception):
        fs._validate_kwargs(dict.fromkeys(["x", "y"], 0), valid=["x"], mandatory=["x"])
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
        fs._validate_kwargs(dict.fromkeys(["x", "y"], 0), valid=["x", "y", "z"]) is None
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


def test_namespaces(fs, filesystem):
    print("Testing namespaces...")

    ns1 = random_string(5)
    ns2 = random_string(5)

    # Namespace without url should raise exception
    with pytest.raises(Exception):
        fs.create_namespace(ns1, description="ns1")

    fs.create_namespace(
        ns1,
        description="ns1",
        url=posixpath.join(build_url(filesystem), ns1),
        storage_options=filesystem["fs"].storage_options,
        backend=filesystem["backend"],
    )

    # Create with duplicate url should raise exception
    with pytest.raises(Exception):
        fs.create_namespace(
            ns2,
            description="ns2",
            url=posixpath.join(build_url(filesystem), ns1),
            storage_options=filesystem["fs"].storage_options,
            backend=filesystem["backend"],
        )

    fs.create_namespace(
        ns2,
        description="ns2",
        url=posixpath.join(build_url(filesystem), ns2),
        storage_options=filesystem["fs"].storage_options,
        backend=filesystem["backend"],
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


def test_features(fs, filesystem):
    print("Testing features...")

    fs.create_feature("test/feature1", description="feature1")
    fs.create_feature("feature2", namespace="test", description="feature2")
    fs.create_namespace(
        "test2",
        description="test2",
        url=posixpath.join(build_url(filesystem), "test2"),
        storage_options=filesystem["fs"].storage_options,
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


def test_data_deletion(fs, filesystem, file_name):
    # Only do this for local data storage
    if filesystem["fs"].__class__.__name__ != "LocalFileSystem":
        pytest.skip("Skipping data deletion tests on cloud storage")
    print("Testing data deletion...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
    df1 = pd.DataFrame(
        {"time": dts, "value": np.random.randint(0, 100, size=len(dts))}
    ).set_index("time")
    fs.create_feature(
        "test/feature-to-delete",
    )
    fs.save_dataframe(df1, "test/feature-to-delete")
    # Check data exists
    assert os.path.isdir(
        posixpath.join(
            filesystem["location"], file_name, "feature", "feature-to-delete"
        )
    )
    fs.delete_feature("test/feature-to-delete", delete_data=True)
    # Data should now have gone
    assert not os.path.isdir(
        posixpath.join(
            filesystem["location"], file_name, "feature", "feature-to-delete"
        )
    )

    fs.create_feature(
        "test/feature-to-delete",
    )
    fs.save_dataframe(df1, "test/feature-to-delete")
    # Check data exists
    assert os.path.isdir(
        posixpath.join(
            filesystem["location"], file_name, "feature", "feature-to-delete"
        )
    )
    fs.delete_feature("test/feature-to-delete")
    # Check data still exists
    assert os.path.isdir(
        posixpath.join(
            filesystem["location"], file_name, "feature", "feature-to-delete"
        )
    )
    # Call clean_namespace to get rid of data
    fs.clean_namespace("test")
    assert not os.path.isdir(
        posixpath.join(
            filesystem["location"], file_name, "feature", "feature-to-delete"
        )
    )


def test_clone_features(fs):
    print("Testing cloned features")

    dts = pd.date_range("2021-01-01", "2021-01-10")
    df1 = pd.DataFrame(
        {"time": dts, "value": np.random.randint(0, 100, size=len(dts))}
    ).set_index("time")
    fs.create_feature("test/old-feature", description="Will be cloned", serialized=True)
    fs.save_dataframe(df1, "test/old-feature")
    fs.clone_feature("test/cloned-feature", from_name="test/old-feature")
    feature = fs.list_features(name="test/cloned-feature").iloc[0]
    # Check that metadata was copied
    assert feature.description == "Will be cloned"
    assert feature.serialized == True
    # Check that data was copied
    result = fs.load_dataframe("test/cloned-feature")
    compare_df(result, df1.rename(columns={"value": "test/cloned-feature"}))

    fs.delete_feature("test/old-feature")
    fs.delete_feature("test/cloned-feature")


def test_dataframes(fs):
    print("Testing data load/save...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
    df1 = pd.DataFrame({"time": dts, "value": np.random.randn(len(dts))}).set_index(
        "time"
    )
    dts = pd.date_range("2021-01-01", "2021-02-01", freq="60min")
    df2 = pd.DataFrame({"time": dts, "value": [{"x": np.random.randn()} for x in dts]})
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
    assert compare_df(
        fs.load_dataframe("test/df1"), df1.rename(columns={"value": "test/df1"})
    )
    assert compare_df(
        fs.load_dataframe("test/df2"),
        df2.set_index("time").rename(columns={"value": "test/df2"}),
    )
    assert compare_df(
        fs.load_dataframe(["test/df3", "test/df4"]), df3.set_index("time")
    )

    # Delete features
    fs.delete_feature("test/df1")
    fs.delete_feature("test/df2")
    fs.delete_feature("test/df3")
    fs.delete_feature("test/df4")


def test_resampling(fs):
    print("Testing data resampling...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
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

    # Pandas tests
    result = fs.load_dataframe(["test/resample1", "test/resample2"])
    compare = pd.concat([df1, df2], join="outer", axis=1).ffill()
    assert compare_df(result, compare)
    result = fs.load_dataframe(["test/resample1", "test/resample2"], freq="2d")
    compare = pd.concat([df1, df2], join="outer", axis=1).resample("2d").ffill().ffill()
    assert compare_df(result, compare)
    result = fs.load_dataframe(["test/resample1", "test/resample2"], freq="10min")
    compare = (
        pd.concat([df1, df2], join="outer", axis=1).resample("10min").ffill().ffill()
    )
    assert compare_df(result, compare)
    result = fs.load_dataframe(
        ["test/resample1", "test/resample2"],
        freq="10min",
        from_date="2021-01-10",
        to_date="2021-01-12",
    )
    compare = (
        pd.concat([df1, df2], join="outer", axis=1).resample("10min").ffill().ffill()
    )
    compare = compare[
        (compare.index >= pd.Timestamp("2021-01-10"))
        & (compare.index <= pd.Timestamp("2021-01-12"))
    ]
    assert compare_df(result, compare)
    # Use dataframe to specify which features to load
    result = fs.load_dataframe(
        fs.list_features(regex=r"resample."),
        freq="10min",
        from_date="2021-01-10",
        to_date="2021-01-12",
    )
    assert compare_df(result, compare)
    result = fs.load_dataframe(
        "test/resample1",
        from_date="2021-01-10",
        to_date="2021-01-12",
    )
    compare = df1[
        (df1.index >= pd.Timestamp("2021-01-10"))
        & (df1.index <= pd.Timestamp("2021-01-12"))
    ]
    assert compare_df(result, compare)

    # Non-contiguous resampling
    dts = pd.date_range("2021-01-01", "2021-01-05")
    df3 = pd.DataFrame(
        {"time": dts, "test/resample3": np.random.randn(len(dts))}
    ).set_index("time")
    dts = pd.date_range("2021-01-10", "2021-02-15")
    df4 = pd.DataFrame(
        {"time": dts, "test/resample4": np.random.randn(len(dts))}
    ).set_index("time")
    fs.create_feature("test/resample3", description="df3")
    fs.create_feature("test/resample4", description="df4")
    fs.save_dataframe(df3)
    fs.save_dataframe(df4)

    compare = pd.concat([df3, df4], join="outer", axis=1).resample("1d").ffill().ffill()
    compare = compare[compare.index >= pd.Timestamp("2021-01-14")]
    result = fs.load_dataframe(
        ["test/resample3", "test/resample4"], from_date="2021-01-14", freq="1d"
    )
    assert compare_df(result, compare)

    fs.delete_feature("test/resample1")
    fs.delete_feature("test/resample2")
    fs.delete_feature("test/resample3")
    fs.delete_feature("test/resample4")


def test_serialized_features(fs):
    print("Testing JSON serialized features")

    fs.create_feature("test/non-serialized")
    fs.create_feature("test/serialized", serialized=True)

    dts = pd.date_range("2020-01-01", "2021-01-01")
    df = pd.DataFrame(
        {
            "time": dts,
            # Values with changing schema
            "value": [idx if idx < 150 else {"x": idx} for idx, x in enumerate(dts)],
        }
    ).set_index("time")

    # Raise exception if trying to change serialzation on existing feature
    with pytest.raises(Exception):
        fs.update_feature("test/non-serialized", serialized=True)
    # Raise exception if schema changes on non-serialized feature
    with pytest.raises(Exception):
        fs.save_dataframe(df, "test/non-serialized")
    # This should work
    fs.save_dataframe(df, "test/serialized")
    result = fs.load_dataframe("test/serialized")
    assert compare_df(result, df.rename(columns={"value": "test/serialized"}))

    fs.delete_feature("test/non-serialized")
    fs.delete_feature("test/serialized")


def test_empty_features(fs):
    print("Testing empty feature datasets...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
    df1 = pd.DataFrame(
        {"time": dts, "test/empty1": np.random.randn(len(dts))}
    ).set_index("time")
    fs.create_feature("test/empty1")

    result = fs.load_dataframe(["test/empty1"])
    assert empty_df(result)
    result = fs.load_dataframe(
        ["test/empty1"], from_date="2021-01-01", to_date="2021-01-10", freq="1d"
    )
    assert len(result) == len(dts)

    fs.save_dataframe(df1)
    # Load data outside of time range
    result = fs.load_dataframe(
        ["test/empty1"], from_date="2020-01-01", to_date="2020-03-01"
    )
    assert empty_df(result)

    fs.delete_feature("test/empty1")


def test_time_travel(fs):
    print("Testing time travel...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
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
    assert compare_df(result, df1.drop(columns="created_time"))
    result = fs.load_dataframe("test/timetravel1", time_travel="-15min")
    assert compare_df(result, df2.drop(columns="created_time"))
    result = fs.load_dataframe("test/timetravel1", time_travel="-60min")
    assert compare_df(result, df3.drop(columns="created_time"))
    result = fs.load_dataframe("test/timetravel1", time_travel="-120min")
    assert empty_df(result)

    fs.delete_feature("test/timetravel1")


def test_last_values(fs):
    print("Testing last feature values...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
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


def test_transforms(fs):
    print("Testing feature transforms...")

    dts = pd.date_range("2021-01-01", "2021-01-10")
    df1 = pd.DataFrame(
        {
            "time": dts,
            "test/raw-feature": np.random.randint(0, 100, size=len(dts)),
        }
    ).set_index("time")

    fs.create_feature("test/raw-feature")
    fs.save_dataframe(df1)

    # Create some transforms
    @fs.transform("test/squared-feature", from_features=["test/raw-feature"])
    def square(df):
        return df ** 2

    @fs.transform(
        "test/combined-feature",
        from_features=["test/raw-feature", "test/squared-feature"],
    )
    def add(df):
        return df["test/raw-feature"] + df["test/squared-feature"]

    # Get transformed features
    result = fs.load_dataframe(
        ["test/raw-feature", "test/squared-feature", "test/combined-feature"]
    )
    assert compare_series(
        result["test/squared-feature"], result["test/raw-feature"] ** 2
    )
    assert compare_series(
        result["test/combined-feature"],
        result["test/raw-feature"] ** 2 + result["test/raw-feature"],
    )

    result = fs.last(
        ["test/raw-feature", "test/squared-feature", "test/combined-feature"]
    )
    assert result["test/squared-feature"] == result["test/raw-feature"] ** 2
    assert (
        result["test/combined-feature"]
        == result["test/raw-feature"] ** 2 + result["test/raw-feature"]
    )

    # Try to create recursive feature loop
    fs.create_feature("test/recursive-feature")

    @fs.transform("test/recursive-feature-2", from_features=["test/recursive-feature"])
    def passthrough(df):
        return df

    @fs.transform("test/recursive-feature", from_features=["test/recursive-feature-2"])
    def passthrough(df):
        return df

    # This should fail
    with pytest.raises(Exception):
        df = fs.load_dataframe("test/recursive-feature")
    with pytest.raises(Exception):
        df = fs.load_dataframe("test/recursive-feature-2")

    fs.delete_feature("test/raw-feature")
    fs.delete_feature("test/squared-feature")
    fs.delete_feature("test/combined-feature")
    fs.delete_feature("test/recursive-feature")
    fs.delete_feature("test/recursive-feature-2")
