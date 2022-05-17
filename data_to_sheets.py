from functools import partial
import json
import os
import pandas as pd
import gspread
from df2gspread import df2gspread as d2g
import pygsheets

script_dir = os.path.dirname(__file__)
rel_path = "cbscraper/big_data.jsonlines"
data_path = os.path.join(script_dir, rel_path)


def load_data() -> pd.DataFrame:
    with open(data_path, "rb") as f:
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ["json_element"]
        df_inter["json_element"].apply(json.loads)
        df = pd.json_normalize(df_inter["json_element"].apply(json.loads))
    df.drop_duplicates(subset=["url"], inplace=True)
    df.product_name = list(
        map(
            partial(" - ".join),
            zip(
                df["brand"].values,
                df["product_name"].values,
            ),
        )
    )

    return df


from google.cloud import storage

df = load_data()


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client.from_service_account_json("client_secret.json")
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name, timeout=1000)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

df.to_pickle('data.pkl')

bucket_name = "sephora_scraped_data"

upload_blob(bucket_name, "data.pkl", "data.pkl")
