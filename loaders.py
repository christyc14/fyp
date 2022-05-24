from __future__ import print_function
from pathlib import Path
from tempfile import NamedTemporaryFile
import streamlit as st
import pandas as pd

from autogluon.tabular import TabularPredictor

from google.cloud import storage


@st.experimental_memo
def load_data() -> pd.DataFrame:
    storage_client = storage.Client()
    bucket = storage_client.bucket("sephora_scraped_data")
    blob = bucket.blob("data.pkl")
    with NamedTemporaryFile() as f:
        blob.download_to_filename(f.name)
        return pd.read_pickle(f.name)


@st.experimental_memo
def load_model() -> TabularPredictor:
    storage_client = storage.Client()
    bucket = storage_client.bucket("sephora_scraped_data")
    blobs = bucket.list_blobs(prefix="agModels-predictClass")
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        file_split = blob.name.split("/")
        directory = "/".join(file_split[0:-1])
        Path(directory).mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(blob.name)
    return TabularPredictor.load("agModels-predictClass/")
