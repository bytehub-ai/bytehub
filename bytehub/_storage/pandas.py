from .dask import Store as Dask


class Store(Dask):
    """Pandas-backed timeseries data storage (uses Dask for parquet functions)."""

    def __init__(self):
        super(Store, self).__init__()

    def load(
        self, name, from_date=None, to_date=None, freq=None, time_travel=None, **kwargs
    ):
        ddf = self._read(name, from_date, to_date, freq, time_travel, **kwargs)
        if not from_date:
            from_date = ddf.index.min().compute()  # First value in data
        if not to_date:
            to_date = ddf.index.max().compute()  # Last value in data
        pdf = ddf.compute()
        # Keep only last created_time for each index timestamp
        pdf = (
            pdf.reset_index()
            .set_index("created_time")
            .sort_index()
            .groupby("time")
            .last()
        )
        # Apply resampling/date filtering
        if freq:
            samples = pd.DataFrame(index=pd.date_range(from_date, to_date, freq=freq))
            pdf = pd.merge(
                pd.merge(
                    pdf, samples, left_index=True, right_index=True, how="outer"
                ).ffill(),
                samples,
                left_index=True,
                right_index=True,
                how="inner",
            )
        else:
            # Filter on date range
            pdf = pdf.loc[pd.Timestamp(from_date) : pd.Timestamp(to_date)]
        if "partition" in pdf.columns:
            pdf = pdf.drop(columns="partition")
        return pdf

    def transform(self, df, func):
        transformed = func(df)
        # Make sure output has a single column named 'value'
        if isinstance(transformed, pd.Series) or isinstance(transformed, dd.Series):
            transformed = transformed.to_frame("value")
        if not isinstance(transformed, pd.DataFrame):
            raise RuntimeError(
                f"This featurestore has a Pandas backend, therefore transforms should return Pandas dataframes"
            )
        if len(transformed.columns) != 1:
            raise RuntimeError(
                f"Transform function should return a dataframe with a datetime index and single value column"
            )
        transformed.columns = ["value"]
        return transformed
