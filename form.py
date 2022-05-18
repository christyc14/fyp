from __future__ import print_function

from functools import partial
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import streamlit as st
import os
import pandas as pd
import numpy as np
from calendar import c
from typing import Dict, List
from zlib import DEF_BUF_SIZE
import json
import smtplib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.sparse.linalg import svds
from scipy.spatial import distance
from autogluon.tabular import TabularPredictor
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import base64
import time
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from pprint import pprint
from google.cloud import storage
from svd import content_recommender
from svd import collab_recommender

script_dir = os.path.dirname(__file__)
rel_path = "cbscraper/big_data.jsonlines"
data_path = os.path.join(script_dir, rel_path)


def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">Sephora US Link</a>'


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
        return TabularPredictor.load('agModels-predictClass/')

df: pd.DataFrame = load_data()
product_categories: List[str] = list(set(df.category.values))
fields_to_csv = {}


def follow_up_questions(cbf, cat, fields_to_csv):
    have_tried = st.radio(
        "Have you tried any of these products?",
        options=["Yes", "No"],
        key=f"{cat}_tried",
    )
    fields_to_csv[f"have_tried_svd_{cat}"] = have_tried
    if have_tried == "Yes":
        st.write("Which of these have you tried?")
        tried_products = [
            st.checkbox(item.product_name, key=f"{cat}_tried_products")
            for _, item in cbf.iterrows()
        ]
        tried_products_svd = [
            row.product_name
            for k, (_, row) in enumerate(cbf.iterrows())
            if tried_products[k]
        ]
        fields_to_csv[f"tried_products_svd_{cat}"] = tried_products_svd
    else:
        tried_products_svd = []
    willing = st.radio(
        "Would you be willing to try any of these products?",
        key=f"{cat}_willing",
        options=["Yes", "No"],
    )
    fields_to_csv[f"willing_svd_{cat}"] = willing
    if willing == "Yes":
        willing_products_temp = []
        for _, item in cbf.iterrows():
            if item.product_name not in tried_products_svd:
                willing_products_temp.append(item.product_name)
        willing_products = [
            st.checkbox(item, key=f"_{item}") for item in willing_products_temp
        ]
        will_prod_names = [
            item
            for j, item in enumerate(willing_products_temp)
            if willing_products[j]
        ]
        fields_to_csv[f"will_prod_names_svd_{cat}"] = will_prod_names


def svd(df, product_categories, fields_to_csv):
    st.write("Which categories would you like recommendations for?")
    checkboxes = [st.checkbox(cat, value=False) for cat in product_categories]
    cat_svd = [cat for i, cat in enumerate(product_categories) if checkboxes[i]]
    fields_to_csv["category"] = cat_svd
    top_3: Dict[str, List[str]] = {}
    if any(checkboxes):
        st.write("Which are your top 3 products for each category?")
        for i, checkbox in enumerate(checkboxes):
            if checkbox:
                top_3[product_categories[i]] = st.multiselect(
                    product_categories[i],
                    df[df.category == product_categories[i]]["product_name"].values,
                )
        if any([len(v) > 3 for v in top_3.values()]):
            st.warning("Please select at most 3 products for each category")
        elif all([len(v) > 0 for v in top_3.values()]):
            fields_to_csv["top3"] = top_3
            user_has_sephora: bool = (
                st.radio(
                    "Do you have a Sephora (US) account?",
                    options=["Yes", "No"],
                    index=1,
                )
                == "Yes"
            )
            if user_has_sephora:
                fields_to_csv[f"user_has_sephora"] = user_has_sephora
                multiple_reviews: bool = (
                    st.radio(
                        "Have you written 2 or more reviews?", options=["Yes", "No"]
                    )
                    == "Yes"
                )
            if user_has_sephora and multiple_reviews:
                fields_to_csv["multiple_reviews"] = multiple_reviews
                reviews = df.explode("review_data")
                reviews["username"] = reviews["review_data"].apply(
                        lambda x: x["UserNickname"]
                    )
                reviews["rating"] = reviews["review_data"].apply(
                        lambda x: x["Rating"]
                    )
                grouped_reviews = reviews.groupby("username")["review_data"].apply(
                        list
                    )
                multiple_rating_users = [""] + list(
                    set(grouped_reviews[grouped_reviews.map(len) > 1].index)
                )
                username = st.selectbox(
                    "Please select your username from dropdown",
                    sorted(multiple_rating_users),
                )
                fields_to_csv["username"] = username
                user_not_in_list = st.checkbox(
                    "Please check here if your username is not in this list",
                    key="user_not_in_list",
                )
                if user_not_in_list:
                    user_has_sephora = False
                for cat, products in top_3.items():
                    products = (
                        (
                            *products,
                            *((3 - (len(products))) * (products[len(products) - 1],)),
                        )
                        if len(products) < 3
                        else products
                    )
                    cbf = content_recommender(
                        cat,
                        *(products),
                        15,
                        df,
                    )
                    reviews_cbf = cbf.explode("review_data")
                    reviews_cbf["username"] = reviews_cbf["review_data"].apply(
                            lambda x: x["UserNickname"]
                        )
                    reviews_cbf["rating"] = reviews_cbf["review_data"].apply(
                            lambda x: x["Rating"]
                        )
                    grouped_reviews_cbf = reviews_cbf.groupby("username")["review_data"].apply(
                            list
                        )
                    multiple_rating_users_cbf = [""] + list(
                        set(grouped_reviews_cbf[grouped_reviews_cbf.map(len) > 1].index)
                    )
                    if username in multiple_rating_users_cbf:
                        st.write(f"Here are your {cat} recommendations:")
                        cf = collab_recommender(cbf, 5, username)
                        fields_to_csv[f"recs_{cat}"] = cf[["product_name"]].to_json()
                        disp = cf[["brand", "product_name", "url"]]
                        disp["url"] = disp["url"].apply(make_clickable)
                        disp = disp[["brand", "product_name", "url"]].to_html(
                            escape=False
                        )
                        st.write(disp, unsafe_allow_html=True)
                        follow_up_questions(cf, cat, fields_to_csv)
                    else:
                        st.write(f"Here are your {cat} recommendations:")
                        fields_to_csv[f"recs_{cat}"] = cbf[["product_name"]].to_json()
                        disp = cbf[["brand", "product_name", "url"]].head(5)
                        disp["url"] = disp["url"].apply(make_clickable)
                        disp = disp[["brand", "product_name", "url"]].to_html(
                            escape=False
                        )
                        st.write(disp, unsafe_allow_html=True)
                        follow_up_questions(cbf.head(5), cat, fields_to_csv)
                        
            if not user_has_sephora or not multiple_reviews:  # if no sephora acc
                fields_to_csv["user_has_sephora"] = user_has_sephora
                print(top_3)
                for cat, products in top_3.items():
                    st.write(f"Here are your {cat} recommendations:")
                    products = (
                        (
                            *products,
                            *((3 - (len(products))) * (products[len(products) - 1],)),
                        )
                        if len(products) < 3
                        else products
                    )

                    cbf = content_recommender(
                        cat,
                        *(products),
                        5,
                        df,
                    ).head(5)
                    # link is the column with hyperlinks
                    disp = cbf[["brand", "product_name", "url"]]
                    fields_to_csv[f"recs_{cat}"] = disp.to_json()
                    disp["url"] = disp["url"].apply(make_clickable)
                    disp = disp[["brand", "product_name", "url"]].to_html(escape=False)
                    st.write(disp, unsafe_allow_html=True)
                    follow_up_questions(cbf.head(5), cat, fields_to_csv)

            reccommend = st.slider(
                "How likely would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
                min_value=0,
                max_value=10,
                value=1,
            )
            fields_to_csv["nps_score"] = reccommend
            nps_reason = st.text_input("Please explain why you gave that score.")
            fields_to_csv["nps_reason"] = nps_reason
            return nps_reason != ""


def ml(df, product_categories, fields):
    ml_pred = load_model()
    # ask questions
    st.write("Which categories would you like recommendations for?")
    checkboxes_ml = [
        st.checkbox(cat, value=False, key=f"{cat}_ml") for cat in product_categories
    ]
    cat_ml = [cat for i, cat in enumerate(product_categories) if checkboxes_ml[i]]
    fields["cat_ml"] = cat_ml
    skin_tone = "fair"
    skin_type = "dry"
    skin_concern = "acne"
    nps_reason_ml = ""
    if any(checkboxes_ml):
        skin_tone = st.selectbox(
            "What is your skin tone?",
            options=[
                "",
                "porcelain",
                "fair",
                "light",
                "medium",
                "tan",
                "olive",
                "deep",
                "dark",
                "ebony",
            ],
            key="skin_tone",
        )
        fields["skin_tone"] = skin_tone
        if skin_tone != "":
            skin_type = st.selectbox(
                "What is your skin type?",
                options=["", "normal", "combination", "dry", "oily"],
                key="skin_type",
            )
            fields["skin_type"] = skin_type
            if skin_type != "":
                skin_concern = st.selectbox(
                    "What are your most important skincare concern?",
                    options=[
                        "",
                        "acne",
                        "aging",
                        "blackheads",
                        "darkCircles",
                        "dullness",
                        "sensitivity",
                        "redness",
                        "sunDamage",
                        "cellulite",
                        "pores",
                        "unevenSkinTones",
                        "cuticles",
                        "calluses",
                        "stretchMarks",
                        "puffiness",
                    ],
                    key="skin_concern",
                )
                fields["skin_concern"] = skin_concern
    if skin_tone != "" and skin_type != "" and skin_concern != "":
        for cat in cat_ml:
            ml_df = pd.DataFrame()
            ml_df["ingredients"] = df["ingredients"]
            ml_df["category"] = df["category"]
            ml_df["skin_tone"] = skin_tone
            ml_df["skin_type"] = skin_type
            ml_df["skin_concern"] = skin_concern
            y_pred = ml_pred.predict(ml_df)
            df_tmp = df.copy()
            df_tmp["pred_rating"] = y_pred
            df_tmp = df_tmp.sort_values(by="pred_rating", ascending=False)
            df_tmp = df_tmp[df_tmp["category"] == cat].head(5)
            fields[f"recs_ml_{cat}"] = df_tmp[
                ["brand", "product_name", "pred_rating"]
            ].to_json()
            ml_disp = df_tmp[["brand", "product_name", "url"]]
            ml_disp["url"] = ml_disp["url"].apply(make_clickable)
            ml_disp = ml_disp[["brand", "product_name", "url"]].to_html(escape=False)
            st.write(ml_disp, unsafe_allow_html=True, key="ml_recommendations")
            have_tried_ml = st.radio(
                "Have you tried any of these products?",
                options=["Yes", "No"],
                key=f"{cat}_tried_ml",
            )
            fields[f"{cat}_tried_ml"] = have_tried_ml
            if have_tried_ml == "Yes":
                st.write("Which of these have you tried?")
                tried_products_ml_bool = [
                    st.checkbox(item.product_name, key=f"{cat}_tried_products_ml")
                    for _, item in df_tmp.iterrows()
                ]
                tried_products_ml = []
                for i, item in enumerate(df_tmp.iterrows()):
                    if tried_products_ml_bool[i]:
                        tried_products_ml.append(item)
                fields[f"{cat}_tried_products_ml"] = tried_products_ml
            else:
                tried_products_ml = []
            willing_ml = st.radio(
                "Would you be willing to try any of these products?",
                key=f"{cat}_willing_ml",
                options=["Yes", "No"],
            )
            fields[f"{cat}_willing_ml"] = willing_ml
            if willing_ml == "Yes":
                willing_products_temp_ml = []
                for _, item in df_tmp.iterrows():
                    if item.product_name not in tried_products_ml:
                        willing_products_temp_ml.append(item.product_name)
                willing_products_ml_bool = [
                    st.checkbox(item, key=f"{item}_willing_ml")
                    for item in willing_products_temp_ml
                ]
                willing_products_ml = [ item for i, item in enumerate(willing_products_temp_ml) if willing_products_ml_bool[i] ]
                fields[f"{cat}_willing_products_ml"] = willing_products_ml

        reccommend_ml = st.slider(
            "How likely would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
            min_value=0,
            max_value=10,
            value=1,
            key="ml_reccommend",
        )
        fields["ml_reccommend"] = reccommend_ml
        nps_reason_ml = st.text_input(
            "Please explain why you gave that score.", key="ml_nps_reason"
        )
        fields["ml_nps_reason"] = nps_reason_ml
    return nps_reason_ml != ""


def send_email(fields):
    configuration = sib_api_v3_sdk.Configuration()
    configuration.api_key[
        "api-key"
    ] = "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"
    attachment = fields
    # create an instance of the API class
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(
        sib_api_v3_sdk.ApiClient(configuration)
    )
    subject = "My Subject"
    html_content = f"<html><body><h1>This is the data </h1>{attachment}</body></html>"
    sender = {"name": "Christy Chan", "email": "christychanseewai@gmail.com"}
    to = [{"email": "christychanseewai@gmail.com", "name": "Jane Doe"}]
    reply_to = {"email": "replyto@domain.com", "name": "John Doe"}
    headers = {"Some-Custom-Name": "unique-id-1234"}
    params = {"parameter": "My param value", "subject": "New Subject"}
    send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(
        to=to,
        reply_to=reply_to,
        headers=headers,
        html_content=html_content,
        sender=sender,
        subject=subject,
    )

    try:
        # Send a transactional email
        api_response = api_instance.send_transac_email(send_smtp_email)
        pprint(api_response)
        st.write("Responses have been saved!")
        st.session_state["email"] = "sent"
    except ApiException as e:
        print("Exception when calling SMTPApi->send_transac_email: %s\n" % e)


if "k" not in st.session_state:
    st.session_state["k"] = random.random()

if "email" not in st.session_state:
    st.session_state["email"] = ""

st.title("FYP Questionnaire")
st.caption(
    "Please answer the following questions. There are three parts, one for Recommender A, and one for Recommender B, and one to compare the two."
)
if st.session_state.k < 0.5:
    if svd(df, product_categories, fields_to_csv):
        st.write("Thank you for completing part 1 :)")
        st.write("Part 2 is a different recommender.")
        if ml(df, product_categories, fields_to_csv):
            st.write("Thank you for completing part 2 :)")
            st.write("Part 3 contains questions comparing part 1 and part 2.")
            user_pref = st.radio(
                "Which of the two did you prefer?",
                options=["Part 1", "Part 2", "Hated both :("],
            )
            fields_to_csv["user_pref"] = user_pref
            if user_pref == "Part 1" or "Part 2":
                why_user_pref = st.text_input(f"Why did you prefer {user_pref}?")
                fields_to_csv["why_user_pref"] = why_user_pref
                user_consistent = st.radio(
                    "Would you use either of them consistently?", options=["Yes", "No"]
                )
                fields_to_csv["user_consistent"] = user_consistent
                if user_consistent == "Yes":
                    user_when = st.text_input("When would you use it?")
                    fields_to_csv["user_when"] = user_when
                else:
                    user_not_use = st.text_input("Why not?")
                    fields_to_csv["user_not_use"] = user_not_use
            else:
                why_user_pref = st.text_input("Why did you hate both of them?")
                fields_to_csv["why_user_pref"] = why_user_pref
            if st.button("Submit"):
                if st.session_state.email == "":
                    send_email(fields_to_csv)
                    st.write("Thank you for completing this questionnaire :)")

else:
    if ml(df, product_categories, fields_to_csv):
        st.write("Thank you for completing part 1 :)")
        st.write("Part 2 is a different recommender.")
        if svd(df, product_categories, fields_to_csv):
            st.write("Thank you for completing part 2 :)")
            st.write("Part 3 contains questions comparing part 1 and part 2.")
            user_pref = st.radio(
                "Which of the two did you prefer?",
                options=["Part 1", "Part 2", "Hated both :("],
            )
            fields_to_csv["user_pref"] = user_pref
            if user_pref == "Part 1" or "Part 2":
                why_user_pref = st.text_input(f"Why did you prefer {user_pref}?")
                fields_to_csv["why_user_pref"] = why_user_pref
                user_consistent = st.radio(
                    "Would you use either of them consistently?", options=["Yes", "No"]
                )
                fields_to_csv["user_consistent"] = user_consistent
                if user_consistent == "Yes":
                    user_when = st.text_input("When would you use it?")
                    fields_to_csv["user_when"] = user_when
                else:
                    user_not_use = st.text_input("Why not?")
                    fields_to_csv["user_not_use"] = user_not_use
            else:
                why_user_pref = st.text_input("Why did you hate both of them?")
                fields_to_csv["why_user_pref"] = why_user_pref
            if st.button("Submit"):
                if st.session_state.email == "":
                    send_email(fields_to_csv)
                    st.write("Thank you for completing this questionnaire :)")
