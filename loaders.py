from __future__ import print_function
from pathlib import Path
from tempfile import NamedTemporaryFile
import streamlit as st
import pandas as pd
import pyarrow as pa
from autogluon.tabular import TabularPredictor

@st.experimental_singleton
def load_data() -> pd.DataFrame:
    source = pa.memory_map('data.arrow', 'r')
    table = pa.ipc.RecordBatchFileReader(source).read_all()
    return table.to_pandas()


@st.experimental_singleton
def load_model() -> TabularPredictor:
    return TabularPredictor.load("agModels-predictClass/")
