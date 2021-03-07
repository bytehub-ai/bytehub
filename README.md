# ByteHub [![PyPI Latest Release](https://img.shields.io/pypi/v/bytehub.svg)](https://pypi.org/project/bytehub/) [![Issues](https://img.shields.io/github/workflow/status/bytehub-ai/bytehub/Tests)](https://github.com/bytehub-ai/bytehub/actions?query=workflow%3ATests) [![Issues](https://img.shields.io/github/issues/bytehub-ai/bytehub)](https://github.com/bytehub-ai/bytehub/issues) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://uploads-ssl.webflow.com/5f187c12c1b99c41557b035e/6026e99dad5c3cf816547670_bytehub-rect-logo.png" align="right" alt="ByteHub logo" width="120" height="60">

An easy-to-use feature store.



## 💾 What is a feature store?

A feature store is a data storage system for data science and machine-learning. It can store _raw data_ and also transformed _features_, which can be fed straight into an ML model or training script.

Feature stores allow data scientists and engineers to be **more productive** by organising the flow of data into models.

The [Bytehub Feature Store](https://www.bytehub.ai) is designed to:
* Be simple to use, with a Pandas-like API;
* Require no complicated infrastructure, running on a local Python installation or in a cloud environment;
* Be optimised towards timeseries operations, making it highly suited to applications such as those in finance, energy, forecasting; and
* Support simple time/value data as well as complex structures, e.g. dictionaries.

It is built on [Dask](https://dask.org/) to support large datasets and cluster compute environments.

## 🦉 Features

* Searchable **feature information** and **metadata** can be stored locally using SQLite or in a remote database.
* Timeseries data is saved in [Parquet format](https://parquet.apache.org/) using Dask, making it readable from a wide range of other tools. Data can reside either on a local filesystem or in a [cloud storage service](https://docs.dask.org/en/latest/remote-data-services.html), e.g. AWS S3. 
* Supports **timeseries joins**, along with **filtering** and **resampling** operations to make it easy to load and prepare datasets for ML training.
* Feature engineering steps can be implemented as **transforms**. These are saved within the feature store, and allows for simple, resusable preparation of raw data.
* **Time travel** can retrieve feature values based on when they were created, which can be useful for forecasting applications.
* Simple APIs to retrieve timeseries dataframes for training, or a dictionary of the most recent feature values, which can be used for inference.

Also available as **[☁️ ByteHub Cloud](https://bytehub.ai)**: a ready-to-use, cloud-hosted feature store.

## 📖 Documentation and tutorials

See the [ByteHub documentation](https://docs.bytehub.ai/) and [notebook tutorials](https://github.com/bytehub-ai/code-examples/tree/main/tutorials) to learn more and get started.

## 🚀 Quick-start

Install using pip:

```sh
pip install bytehub
```

Create a local SQLite feature store by running:

```python
import bytehub as bh
import pandas as pd

fs = bh.FeatureStore()
```

Data lives inside _namespaces_ within each feature store. They can be used to separate projects or environments. Create a namespace as follows:

```python
fs.create_namespace(
    'tutorial', url='/tmp/featurestore/tutorial', description='Tutorial datasets'
)
```

Create a _feature_ inside this namespace which will be used to store a timeseries of pre-prepared data:

```python
fs.create_feature('tutorial/numbers', description='Timeseries of numbers')
```

Now save some data into the feature store:

```python
dts = pd.date_range('2020-01-01', '2021-02-09')
df = pd.DataFrame({'time': dts, 'value': list(range(len(dts)))})

fs.save_dataframe(df, 'tutorial/numbers')
```

The data is now stored, ready to be transformed, resampled, merged with other data, and fed to machine-learning models.

We can engineer new features from existing ones using the _transform_ decorator. Suppose we want to define a new feature that contains the squared values of `tutorial/numbers`:

```python
@fs.transform('tutorial/squared', from_features=['tutorial/numbers'])
def squared_numbers(df):
    # This transform function receives dataframe input, and defines a transform operation
    return df ** 2 # Square the input
```

Now both features are saved in the feature store, and can be queried using:

```python
df_query = fs.load_dataframe(
    ['tutorial/numbers', 'tutorial/squared'],
    from_date='2021-01-01', to_date='2021-01-31'
)
```

To connect to ByteHub Cloud, first [register for an account](https://www.bytehub.ai/feature-store/request-access), then use:

```python
fs = bh.FeatureStore("https://api.bytehub.ai")
```

This will allow you to store features in your own private namespace on ByteHub Cloud, and save datasets to an AWS S3 storage bucket.

## 🐾 Roadmap

* _Tasks_ to automate updates to features using orchestration tools like [Airflow](https://airflow.apache.org/)
