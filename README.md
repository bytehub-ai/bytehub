# ByteHub Feature Store

[![PyPI Latest Release](https://img.shields.io/pypi/v/bytehub.svg)](https://pypi.org/project/bytehub/)
[![Issues](https://img.shields.io/github/workflow/status/bytehub-ai/bytehub/Tests)](https://github.com/bytehub-ai/bytehub/actions?query=workflow%3ATests)
[![Issues](https://img.shields.io/github/issues/bytehub-ai/bytehub)](https://github.com/bytehub-ai/bytehub/issues)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## What is a feature store?

A feature store is a data storage system for data science and machine-learning. It can store _raw data_ and also transformed _features_, which can be fed straight into an ML model or training script.

Feature stores allow data scientists and engineers to be **more productive** by organising the flow of data into models.

The [Bytehub Feature Store](https://www.bytehub.ai) is designed to:
* Be simple to use, with a Pandas-like API;
* Require no complicated infrastructure, running on a local Python installation or in a cloud environment;
* Be optimised towards timeseries operations, making it highly suited to applications such as those in finance, energy, forecasting; and
* Support simple time/value data as well as complex structures, e.g. dictionaries.

It is built on [Dask](https://dask.org/) to support large datasets and cluster compute environments.

## Quick-start

Install using pip:

    pip install bytehub

Create a local SQLite feature store by running:

    import bytehub as bh
    import pandas as pd

    fs = bh.FeatureStore()

Data lives inside _namespaces_ within each feature store. They can be used to separate projects or environments. Create a namespace as follows:

    fs.create_namespace(
        'dev', url='/tmp/featurestore/dev', description='Dev datasets'
    )

Create a _feature_ inside this namespace which will be used to store a timeseries of pre-prepared data:

    fs.create_feature('dev/first-deature', description='First feature')

Finally save some data into the feature store:

    dts = pd.date_range('2020-01-01', '2021-02-09')
    df = pd.DataFrame({'time': dts, 'value': list(range(len(dts)))})

    fs.save_dataframe(df, 'dev/first-deature')

The data is now stored, ready to be resampled, merged with other data, and fed to machine-learning models.

## Roadmap

* _Tasks_ to automate updates to features using orchestration tools like [Airflow](https://airflow.apache.org/)
* _Transforms_ to automate feature engineering activity.

## Documentation and tutorials

Stay tuned... coming soon.