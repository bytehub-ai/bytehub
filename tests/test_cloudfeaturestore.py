import bytehub as bh
import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import random
import string
import pytest
import s3fs
from urllib.parse import urlparse


def random_string(n):
    return "".join(random.choice(string.ascii_lowercase) for x in range(n))


class TestCloudFeatureStore:
    def setup_class(self):
        if "BYTEHUB_TOKEN" not in os.environ:
            pytest.skip(
                "Skipping cloud feature store tests: no sign-in credentials",
                allow_module_level=True,
            )
        else:
            print("Starting tests: cloud feature store")
        # Connect to feature store
        try:
            self.fs = bh.FeatureStore("https://api.dev.bytehub.ai")
            # Store the namespace
            self.namespace = self.fs.list_namespaces().iloc[0].to_dict()["name"]
        except Exception as e:
            pytest.skip(
                "Skipping cloud feature store tests: ByteHub service is down",
                allow_module_level=True,
            )
        # Make sure we start with a clean feature store
        fs = self.fs
        # Delete all features
        features = fs.list_features()
        for f in features.name.tolist():
            fs.delete_feature(name=f, namespace=self.namespace)
        # Clean all namespaces
        namespaces = fs.list_namespaces()
        for idx, ns in namespaces.iterrows():
            fs.clean_namespace(ns["name"])

    def teardown_class(self):
        fs = self.fs
        print("Finished tests: tearing down...")
        # Delete all features
        features = fs.list_features()
        for f in features.name.tolist():
            fs.delete_feature(name=f, namespace=self.namespace)
        # Clean all namespaces
        namespaces = fs.list_namespaces()
        for idx, ns in namespaces.iterrows():
            fs.clean_namespace(ns["name"])

    def test_permissions(self):
        print("Testing permissions...")
        fs = self.fs

        # Should only have access to one namespace
        namespaces = fs.list_namespaces()
        assert len(namespaces) == 1

        # All features should belong to this namespace
        features = fs.list_features()
        if not features.empty:
            assert len(features.namespace.unique()) == 1
            assert self.namespace in features.namespace.unique()

        # Should not be able to list folders in S3 bucket
        so = fs.list_namespaces().iloc[0].storage_options
        so = {k: v for k, v in so.items() if not k.startswith("_")}
        s3 = s3fs.S3FileSystem(**so)
        bucket = urlparse(namespaces.iloc[0].url).netloc
        with pytest.raises(Exception):
            s3.ls(bucket)

        # Should not be able to update namespace
        with pytest.raises(Exception):
            fs.update_namespace(bucket)

        # Should not be able to create another namespace
        with pytest.raises(Exception):
            fs.create_namespace(name=random_string(), url="s3://whatever")

        # Should not be able to create a feature in another namespace
        with pytest.raises(Exception):
            fs.create_feature(name=f"{random_string()}/{random_string()}")

    def test_features(self):
        print("Testing features...")
        fs = self.fs

        fs.create_feature(f"{self.namespace}/feature1", description="feature1")
        fs.create_feature("feature2", namespace=self.namespace, description="feature2")

        # Duplicate feature
        with pytest.raises(Exception):
            fs.create_feature("{self.namespace}/feature1")
        with pytest.raises(Exception):
            fs.create_feature("feature1", namespace=self.namespace)

        features = fs.list_features(namespace=self.namespace)
        assert "feature1" in features.name.tolist()
        assert "feature2" in features.name.tolist()
        features = fs.list_features(regex="feature.")
        assert len(features) == 2

        fs.delete_feature(f"{self.namespace}/feature1")
        fs.delete_feature("feature2", namespace=self.namespace)
        with pytest.raises(Exception):
            fs.delete_feature("feature2", namespace=self.namespace)

        assert fs.list_features(regex="feature.").empty

    def test_data_deletion(self):
        print("Testing data deletion...")
        fs = self.fs
        namespaces = fs.list_namespaces()
        so = namespaces.iloc[0].storage_options
        so = {k: v for k, v in so.items() if not k.startswith("_")}
        s3 = s3fs.S3FileSystem(**so)
        prefix = urlparse(namespaces.iloc[0].url)
        prefix = prefix.netloc + prefix.path

        dts = pd.date_range("2021-01-01", "2021-01-10")
        df1 = pd.DataFrame(
            {"time": dts, "value": np.random.randint(0, 100, size=len(dts))}
        ).set_index("time")
        fs.create_feature(
            f"{self.namespace}/feature-to-delete",
        )
        fs.save_dataframe(df1, f"{self.namespace}/feature-to-delete")
        # Check data exists
        assert s3.exists(f"{prefix}/feature/feature-to-delete")
        fs.delete_feature(f"{self.namespace}/feature-to-delete", delete_data=True)
        # Data should now have gone
        assert not s3.exists(f"{prefix}/feature/feature-to-delete")

        fs.create_feature(
            f"{self.namespace}/feature-to-delete",
        )
        fs.save_dataframe(df1, f"{self.namespace}/feature-to-delete")
        # Check data exists
        assert s3.exists(f"{prefix}/feature/feature-to-delete")
        fs.delete_feature(f"{self.namespace}/feature-to-delete")
        # Check data still exists
        assert s3.exists(f"{prefix}/feature/feature-to-delete")
        # Call clean_namespace to get rid of data
        fs.clean_namespace(self.namespace)
        assert not s3.exists(f"{prefix}/feature/feature-to-delete")

    def test_clone_features(self):
        print("Testing cloned features")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-01-10")
        df1 = pd.DataFrame(
            {"time": dts, "value": np.random.randint(0, 100, size=len(dts))}
        ).set_index("time")
        fs.create_feature(
            f"{self.namespace}/old-feature",
            description="Will be cloned",
            serialized=True,
        )
        fs.save_dataframe(df1, f"{self.namespace}/old-feature")
        fs.clone_feature(
            f"{self.namespace}/cloned-feature",
            from_name=f"{self.namespace}/old-feature",
        )
        feature = fs.list_features(name=f"{self.namespace}/cloned-feature").iloc[0]
        # Check that metadata was copied
        assert feature.description == "Will be cloned"
        assert feature.serialized == True
        # Check that data was copied
        result = fs.load_dataframe(f"{self.namespace}/cloned-feature")
        assert result.equals(
            df1.rename(columns={"value": f"{self.namespace}/cloned-feature"})
        )

        fs.delete_feature(f"{self.namespace}/old-feature")
        fs.delete_feature(f"{self.namespace}/cloned-feature")

    def test_dataframes(self):
        print("Testing data load/save...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-01-10")
        df1 = pd.DataFrame({"time": dts, "value": np.random.randn(len(dts))}).set_index(
            "time"
        )
        dts = pd.date_range("2021-01-01", "2021-01-10", freq="60min")
        df2 = pd.DataFrame(
            {"time": dts, "value": [{"x": np.random.randn()} for x in dts]}
        )
        df3 = pd.DataFrame(
            {
                "time": dts,
                f"{self.namespace}/df3": np.random.randn(len(dts)),
                f"{self.namespace}/df4": [random_string(5) for x in dts],
            }
        )
        df5 = pd.DataFrame(
            {"time": dts, f"{self.namespace}/df5": np.random.randn(len(dts))}
        )

        # Create features to hold these dataframes
        fs.create_feature(f"{self.namespace}/df1", description="df1")
        fs.create_feature(f"{self.namespace}/df2", description="df2")
        fs.create_feature(f"{self.namespace}/df3", description="df3")
        fs.create_feature(f"{self.namespace}/df4", description="df4", partition="year")

        # Save to non-existent feature
        with pytest.raises(Exception):
            fs.save_dataframe(df1, f"{self.namespace}/df5")
        with pytest.raises(Exception):
            fs.save_dataframe(df5)

        fs.save_dataframe(df1, f"{self.namespace}/df1")
        fs.save_dataframe(df2, "df2", namespace=self.namespace)
        fs.save_dataframe(df3)
        # Try re-writing df1
        fs.save_dataframe(df1, f"{self.namespace}/df1")

        # Load back and check
        assert fs.load_dataframe(f"{self.namespace}/df1").equals(
            df1.rename(columns={"value": f"{self.namespace}/df1"})
        )
        assert fs.load_dataframe(f"{self.namespace}/df2").equals(
            df2.set_index("time").rename(columns={"value": f"{self.namespace}/df2"})
        )
        assert fs.load_dataframe(
            [f"{self.namespace}/df3", f"{self.namespace}/df4"]
        ).equals(df3.set_index("time"))

        # Delete features
        fs.delete_feature(f"{self.namespace}/df1")
        fs.delete_feature(f"{self.namespace}/df2")
        fs.delete_feature(f"{self.namespace}/df3")
        fs.delete_feature(f"{self.namespace}/df4")

    def test_last_values(self):
        print("Testing last feature values...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-01-10")
        df1 = pd.DataFrame(
            {
                "time": dts,
                f"{self.namespace}/last1": np.random.randint(0, 100, size=len(dts)),
                f"{self.namespace}/last2": np.random.randint(0, 100, size=len(dts)),
            }
        ).set_index("time")

        fs.create_feature(f"{self.namespace}/last1")
        fs.create_feature(f"{self.namespace}/last2")
        fs.create_feature(f"{self.namespace}/last3")

        fs.save_dataframe(df1)

        result = fs.last(f"{self.namespace}/last1")
        assert result == {
            f"{self.namespace}/last1": df1[f"{self.namespace}/last1"].values[-1]
        }
        result = fs.last(f"{self.namespace}/last3")
        assert result == {f"{self.namespace}/last3": None}
        result = fs.last(fs.list_features(regex=r"last."))
        assert result == {
            f"{self.namespace}/last1": df1[f"{self.namespace}/last1"].values[-1],
            f"{self.namespace}/last2": df1[f"{self.namespace}/last2"].values[-1],
            f"{self.namespace}/last3": None,
        }

        fs.delete_feature(f"{self.namespace}/last1")
        fs.delete_feature(f"{self.namespace}/last2")
        fs.delete_feature(f"{self.namespace}/last3")

    def test_transforms(self):
        print("Testing feature transforms...")
        fs = self.fs

        dts = pd.date_range("2021-01-01", "2021-01-10")
        df1 = pd.DataFrame(
            {
                "time": dts,
                f"{self.namespace}/raw-feature": np.random.randint(
                    0, 100, size=len(dts)
                ),
            }
        ).set_index("time")

        fs.create_feature(f"{self.namespace}/raw-feature")
        fs.save_dataframe(df1)

        # Create some transforms
        @fs.transform(
            f"{self.namespace}/squared-feature",
            from_features=[f"{self.namespace}/raw-feature"],
        )
        def square(df):
            return df ** 2

        @fs.transform(
            f"{self.namespace}/combined-feature",
            from_features=[
                f"{self.namespace}/raw-feature",
                f"{self.namespace}/squared-feature",
            ],
        )
        def add(df):
            return (
                df[f"{self.namespace}/raw-feature"]
                + df[f"{self.namespace}/squared-feature"]
            )

        # Get transformed features
        result = fs.load_dataframe(
            [
                f"{self.namespace}/raw-feature",
                f"{self.namespace}/squared-feature",
                f"{self.namespace}/combined-feature",
            ]
        )
        assert result[f"{self.namespace}/squared-feature"].equals(
            result[f"{self.namespace}/raw-feature"] ** 2
        )
        assert result[f"{self.namespace}/combined-feature"].equals(
            result[f"{self.namespace}/raw-feature"] ** 2
            + result[f"{self.namespace}/raw-feature"]
        )

        result = fs.last(
            [
                f"{self.namespace}/raw-feature",
                f"{self.namespace}/squared-feature",
                f"{self.namespace}/combined-feature",
            ]
        )
        assert (
            result[f"{self.namespace}/squared-feature"]
            == result[f"{self.namespace}/raw-feature"] ** 2
        )
        assert (
            result[f"{self.namespace}/combined-feature"]
            == result[f"{self.namespace}/raw-feature"] ** 2
            + result[f"{self.namespace}/raw-feature"]
        )

        # Try to create recursive feature loop
        fs.create_feature(f"{self.namespace}/recursive-feature")

        @fs.transform(
            f"{self.namespace}/recursive-feature-2",
            from_features=[f"{self.namespace}/recursive-feature"],
        )
        def passthrough(df):
            return df

        @fs.transform(
            f"{self.namespace}/recursive-feature",
            from_features=[f"{self.namespace}/recursive-feature-2"],
        )
        def passthrough(df):
            return df

        # This should fail
        with pytest.raises(Exception):
            df = fs.load_dataframe(f"{self.namespace}/recursive-feature")
        with pytest.raises(Exception):
            df = fs.load_dataframe(f"{self.namespace}/recursive-feature-2")

        fs.delete_feature(f"{self.namespace}/raw-feature")
        fs.delete_feature(f"{self.namespace}/squared-feature")
        fs.delete_feature(f"{self.namespace}/combined-feature")
        fs.delete_feature(f"{self.namespace}/recursive-feature")
        fs.delete_feature(f"{self.namespace}/recursive-feature-2")
