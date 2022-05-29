from __future__ import print_function
from pathlib import Path
from tempfile import NamedTemporaryFile
import streamlit as st
import pandas as pd
import pyarrow as pa
from autogluon.tabular import TabularPredictor
from google.cloud import storage
import os
from os.path import exists

fname = 'data.arrow'
@st.experimental_singleton
def load_data() -> pd.DataFrame:
    if(os.path.exists(fname)):
        print("loading data from local")
        source = pa.memory_map(fname, 'r')
        table = pa.ipc.RecordBatchFileReader(source).read_all()
        return table.to_pandas()
    else:
        print("loading data from cloud")
        storage_client = storage.Client()
        bucket = storage_client.bucket("sephora_scraped_data")
        blob = bucket.blob(fname)
        blob.download_to_filename(fname)
        source = pa.memory_map(fname, 'r')
        table = pa.ipc.RecordBatchFileReader(source).read_all()
        return table.to_pandas()

@st.experimental_singleton
def load_model() -> TabularPredictor:
    if os.path.isdir("agModels-predictClass/"):
        return TabularPredictor.load("agModels-predictClass/")
    else:
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
