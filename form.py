from __future__ import print_function
from dataclasses import dataclass

from functools import partial
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import streamlit as st
import os
import pandas as pd
import numpy as np
from calendar import c
from typing import Dict, List, Tuple
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

SKINCARE_CONCERNS = [
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
]

SKIN_TONES = [
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
]


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
    return TabularPredictor.load("agModels-predictClass/")


@st.experimental_memo
def get_users_with_multiple_reviews(df) -> List[str]:
    reviews = df.explode("review_data")
    reviews["username"] = reviews["review_data"].apply(lambda x: x["UserNickname"])
    reviews["rating"] = reviews["review_data"].apply(lambda x: x["Rating"])
    grouped_reviews = reviews.groupby("username")["review_data"].apply(list)
    multiple_rating_users = list(
        set(grouped_reviews[grouped_reviews.map(len) > 1].index)
    )
    return [""] + multiple_rating_users


@st.experimental_memo
def pad_selected_products(products: Tuple[str, ...]) -> Tuple[str, ...]:
    return (
        (
            *products,
            *((3 - (len(products))) * (products[len(products) - 1],)),
        )
        if len(products) < 3
        else products
    )


def display_recommendations(df):
    disp = df[["product_name", "url"]]
    disp["url"] = disp["url"].apply(make_clickable)
    disp = disp[["product_name", "url"]].to_html(escape=False)
    st.write(disp, unsafe_allow_html=True)


df: pd.DataFrame = load_data()
predictor = load_model()
product_categories: List[str] = list(set(df.category.values))
fields_to_csv = {}


def follow_up_questions(cbf, cat, fields_to_csv):
    st.write("Have you already tried any of these products? If so, please select them.")
    tried_products = [
        st.checkbox(product_name, key=f"{cat}_tried_products")
        for product_name in cbf["product_name"].values
    ]
    tried_products_svd = cbf["product_name"][tried_products]
    print("tried_productssvd", tried_products_svd)
    fields_to_csv[f"tried_products_svd_{cat}"] = tried_products_svd
    willing = st.radio(
        "Would you be willing to try any of these products?",
        key=f"{cat}_willing",
        options=["Yes", "No"],
        index=1,
    )
    fields_to_csv[f"willing_svd_{cat}"] = willing
    print("willing", willing)
    if willing == "Yes":
        willing_products_temp = cbf["product_name"][
            ~cbf["product_name"].isin(tried_products_svd)
        ]
        willing_products = [
            st.checkbox(item, key=f"_{item}") for item in willing_products_temp
        ]
        will_prod_names = willing_products_temp[willing_products]
        fields_to_csv[f"will_prod_names_svd_{cat}"] = will_prod_names


def category_selection(_):
    st.write("Which categories would you like recommendations for?")
    checkboxes = [st.checkbox(cat, value=False) for cat in product_categories]
    selected_categories = [
        cat for i, cat in enumerate(product_categories) if checkboxes[i]
    ]
    fields_to_csv["category"] = selected_categories
    return selected_categories


def validate_category_selection(selected_categories):
    if len(selected_categories) == 0:
        st.error("Please select at least one category")
        return False
    return True


def top3_product_selection(selected_categories):
    top_3: Dict[str, List[str]] = {}
    st.write("Which are your top 3 products for each category?")
    for category in selected_categories:
        top_3[category] = st.multiselect(
            category,
            df[df.category == category]["product_name"].values,
        )
    return top_3


def validate_top3_product_selection(top_3) -> bool:
    print("hello1")
    if any([len(v) > 3 for v in top_3.values()]):
        st.error("Please select at most 3 products for each category")
        return False
    elif any([len(v) == 0 for v in top_3.values()]):
        st.error("Please select at least 1 product for each category")
        return False
    return True


def user_has_sephora(_):
    st.session_state.user_has_sephora = (
        st.radio(
            "Do you have a Sephora (US) account?",
            options=["Yes", "No"],
            index=1,
        )
        == "Yes"
    )
    return st.session_state.user_has_sephora


def sephora_info(_):
    st.session_state.sephora_username = st.selectbox(
        "Please select your username from dropdown",
        sorted(get_users_with_multiple_reviews(df)),
    )
    st.session_state.sephora_not_in_list = st.checkbox(
        "Please check here if your username is not in this list",
        key="user_not_in_list",
    )


SVD_stages = [
    (category_selection, validate_category_selection),
    (top3_product_selection, validate_top3_product_selection),
    (user_has_sephora, None),
    (sephora_info, None),
]

if "svd_stage_counter" not in st.session_state:
    st.session_state.svd_stage_counter = 0

if "svd_stage_input" not in st.session_state:
    st.session_state.svd_stage_input = None


def svd():
    while st.session_state.svd_stage_counter < len(SVD_stages):
        placeholder = st.empty()
        stage, validator = SVD_stages[st.session_state.svd_stage_counter]
        with placeholder.form(key=f"svd_stage_{st.session_state.svd_stage_counter}"):
            st.session_state.svd_stage_input = stage(st.session_state.svd_stage_input)
            if st.form_submit_button() and (
                validator(st.session_state.svd_stage_input) if validator else True
            ):
                if (
                    stage.__name__ == "user_has_sephora"
                    and not st.session_state.svd_stage_input
                ):
                    st.session_state.svd_stage_counter += 1
                if stage.__name__ == "top3_product_selection":
                    st.session_state.top3 = st.session_state.svd_stage_input
                st.session_state.svd_stage_counter += 1
                placeholder.empty()
            else:
                st.stop()
    if "svd_cat_index" not in st.session_state:
        st.session_state.svd_cat_index = 0
    while st.session_state.svd_cat_index < len(st.session_state.top3.items()):
        cat, products = list(st.session_state.top3.items())[
            st.session_state.svd_cat_index
        ]
        placeholder = st.empty()
        with placeholder.form(key=f"svd_recommendation_{cat}"):
            padded_products = pad_selected_products(products)
            if (
                st.session_state.user_has_sephora
                and not st.session_state.sephora_not_in_list
            ):
                recommendations = content_recommender(
                    cat,
                    *(padded_products),
                    df,
                )
                multiple_rating_users_cbf = get_users_with_multiple_reviews(
                    recommendations
                )
                if (
                    st.session_state.username in multiple_rating_users_cbf
                    and st.session_state.username != ""
                ):
                    recommendations = collab_recommender(
                        recommendations, 5, st.session_state.username
                    )

            else:
                recommendations = content_recommender(
                    cat,
                    *(padded_products),
                    df,
                )

            recommendations = recommendations.head(5)
            fields_to_csv[f"recs_{cat}"] = recommendations[["product_name"]].to_json()
            print("recommendations", recommendations)
            st.write(f"Here are your {cat} recommendations:")
            display_recommendations(recommendations)
            follow_up_questions(recommendations, cat, fields_to_csv)
            if st.form_submit_button():
                st.session_state.svd_cat_index`AZSD` += 1
                placeholder.empty()
            else:
                st.stop()

    with st.form("nps_ml"):
        reccommend = st.slider(
            "How likely would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
            min_value=0,
            max_value=10,
            value=1,
            key="reccommend",
        )
        nps_reason = st.text_input(
            "Please explain why you gave that score.", key="nps_reason"
        )
        if st.form_submit_button():
            st.success("Thanks for your feedback!")
            st.session_state.svd_complete = True


if "ml_stage_counter" not in st.session_state:
    st.session_state.ml_stage_counter = 0

if "ml_stage_input" not in st.session_state:
    st.session_state.ml_stage_input = None

@dataclass
class SkinInfo:
    skin_tone: str
    skin_type: str
    skin_concern: str

def skin_info(_):
    skin_tone = st.selectbox(
        "What is your skin tone?",
        options=SKIN_TONES,
        key="skin_tone",
    )
    skin_type = st.selectbox(
        "What is your skin type?",
        options=["", "normal", "combination", "dry", "oily"],
        key="skin_type",
    )
    skin_concern = st.selectbox(
        "What are your most important skincare concern?",
        options=SKINCARE_CONCERNS,
        key="skin_concern",
    )
    return SkinInfo(skin_tone, skin_type, skin_concern)


ML_stages = [(category_selection, validate_category_selection), (skin_info, None)]


def ml(df, product_categories, fields):
    while st.session_state.ml_stage_counter < len(ML_stages):
        placeholder = st.empty()
        stage, validator = ML_stages[st.session_state.ml_stage_counter]
        with placeholder.form(key=f"ml_stage_{st.session_state.ml_stage_counter}"):
            st.session_state.ml_stage_input = stage(st.session_state.ml_stage_input)
            if st.form_submit_button() and (
                validator(st.session_state.ml_stage_input) if validator else True
            ):
                st.session_state.ml_stage_counter += 1
                placeholder.empty()
            else:
                st.stop()
    if "ml_cat_index" not in st.session_state:
        st.session_state.ml_cat_index = 0
    while st.session_state.ml_cat_index < len(st.session_state.top3.items()):
        cat, products = list(st.session_state.top3.items())[
            st.session_state.ml_cat_index
        ]
        placeholder = st.empty()
        with placeholder.form(key=f"svd_recommendation_{cat}"):
            padded_products = pad_selected_products(products)
            if (
                st.session_state.user_has_sephora
                and not st.session_state.sephora_not_in_list
            ):
                recommendations = content_recommender(
                    cat,
                    *(padded_products),
                    df,
                )
                multiple_rating_users_cbf = get_users_with_multiple_reviews(
                    recommendations
                )
                if (
                    st.session_state.username in multiple_rating_users_cbf
                    and st.session_state.username != ""
                ):
                    recommendations = collab_recommender(
                        recommendations, 5, st.session_state.username
                    )

            else:
                recommendations = content_recommender(
                    cat,
                    *(padded_products),
                    df,
                )

            recommendations = recommendations.head(5)
            fields_to_csv[f"recs_{cat}"] = recommendations[["product_name"]].to_json()
            print("recommendations", recommendations)
            st.write(f"Here are your {cat} recommendations:")
            display_recommendations(recommendations)
            follow_up_questions(recommendations, cat, fields_to_csv)
            if st.form_submit_button():
                st.session_state.cat_index += 1
                placeholder.empty()
            else:
                st.stop()


            # for cat in cat_ml:
            #     ml_df = pd.DataFrame()
            #     ml_df["ingredients"] = df["ingredients"]
            #     ml_df["category"] = df["category"]
            #     ml_df["skin_tone"] = skin_tone
            #     ml_df["skin_type"] = skin_type
            #     ml_df["skin_concern"] = skin_concern
            #     ml_df["product_name"] = df["product_name"]
            #     ml_df["url"] = df["url"]
            #     ml_df["pred_rating"] = ml_pred.predict(
            #         ml_df.drop(["product_name", "url"], axis=1)
            #     )
            #     ml_df = ml_df.sort_values(by="pred_rating", ascending=False)
            #     ml_df = ml_df[ml_df["category"] == cat].head(5)
            #     fields[f"recs_ml_{cat}"] = ml_df[
            #         ["product_name", "pred_rating"]
            #     ].to_json()
            #     print("ml_df", ml_df["product_name"])
            #     display_recommendations(ml_df)
            #     have_tried_ml = st.radio(
            #         "Have you tried any of these products?",
            #         options=["Yes", "No"],
            #         key=f"{cat}_tried_ml",
            #     )
            #     fields[f"{cat}_tried_ml"] = have_tried_ml
            #     print("have_tried_ml", have_tried_ml)
            #     if have_tried_ml == "Yes":
            #         st.write("Which of these have you tried?")
            #         tried_products_ml_bool = [
            #             st.checkbox(product_name, key=f"{cat}_tried_products_ml")
            #             for product_name in ml_df["product_name"].values
            #         ]
            #         tried_products_ml = ml_df["product_name"][tried_products_ml_bool]
            #         print("tried_products_ml", tried_products_ml)
            #         fields[f"{cat}_tried_products_ml"] = tried_products_ml
            #     else:
            #         tried_products_ml = []
            #     willing_ml = st.radio(
            #         "Would you be willing to try any of these products?",
            #         key=f"{cat}_willing_ml",
            #         options=["Yes", "No"],
            #     )
            #     fields[f"{cat}_willing_ml"] = willing_ml
            #     print("willing_ml", willing_ml)
            #     if willing_ml == "Yes":
            #         willing_products_temp_ml = ml_df["product_name"][
            #             ~ml_df["product_name"].isin(tried_products_ml)
            #         ]
            #         willing_products_ml_bool = [
            #             st.checkbox(item, key=f"{item}_willing_ml")
            #             for item in willing_products_temp_ml
            #         ]
            #         willing_products_ml = willing_products_temp_ml[
            #             willing_products_ml_bool
            #         ]
            #         print("willing_products_ml", willing_products_ml)
            #         fields[f"{cat}_willing_products_ml"] = willing_products_ml

            with st.form("nps_ml"):
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
                if st.form_submit_button():
                    st.success("Thanks for your feedback!")


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
    st.session_state["k"] = 0

if "email" not in st.session_state:
    st.session_state["email"] = ""

if "svd_complete" not in st.session_state:
    st.session_state["svd_complete"] = False

st.title("FYP Questionnaire")
st.caption(
    "Please answer the following questions. There are three parts, one for Recommender A, and one for Recommender B, and one to compare the two."
)
if st.session_state.k < 0.5:
    if not st.session_state.svd_complete:
        svd()
    else:
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
            print("user_pref", user_pref)
            if user_pref == "Part 1" or "Part 2":
                why_user_pref = st.text_input(f"Why did you prefer {user_pref}?")
                fields_to_csv["why_user_pref"] = why_user_pref
                print("why_user_pref", why_user_pref)
                user_consistent = st.radio(
                    "Would you use either of them consistently?", options=["Yes", "No"]
                )
                fields_to_csv["user_consistent"] = user_consistent
                print("user_consistent", user_consistent)
                if user_consistent == "Yes":
                    user_when = st.text_input("When would you use it?")
                    fields_to_csv["user_when"] = user_when
                    print("user_when", user_when)
                else:
                    user_not_use = st.text_input("Why not?")
                    fields_to_csv["user_not_use"] = user_not_use
                    print("user_not_use", user_not_use)
            else:
                why_user_pref = st.text_input("Why did you hate both of them?")
                fields_to_csv["why_user_pref"] = why_user_pref
                print("why_user_pref", why_user_pref)
            if st.button("Submit"):
                if st.session_state.email == "":
                    send_email(fields_to_csv)
                    st.write("Thank you for completing this questionnaire :)")

else:
    if ml(df, product_categories, fields_to_csv):
        st.write("Thank you for completing part 1 :)")
        st.write("Part 2 is a different recommender.")
        if svd():
            st.write("Thank you for completing part 2 :)")
            st.write("Part 3 contains questions comparing part 1 and part 2.")
            user_pref = st.radio(
                "Which of the two did you prefer?",
                options=["Part 1", "Part 2", "Hated both :("],
            )
            fields_to_csv["user_pref"] = user_pref
            print("user_pref", user_pref)
            if user_pref == "Part 1" or "Part 2":
                why_user_pref = st.text_input(f"Why did you prefer {user_pref}?")
                fields_to_csv["why_user_pref"] = why_user_pref
                print("why_user_pref", why_user_pref)
                user_consistent = st.radio(
                    "Would you use either of them consistently?", options=["Yes", "No"]
                )
                fields_to_csv["user_consistent"] = user_consistent
                print("user_consistent", user_consistent)
                if user_consistent == "Yes":
                    user_when = st.text_input("When would you use it?")
                    fields_to_csv["user_when"] = user_when
                    print("user_when", user_when)
                else:
                    user_not_use = st.text_input("Why not?")
                    fields_to_csv["user_not_use"] = user_not_use
                    print("user_not_use", user_not_use)
            else:
                why_user_pref = st.text_input("Why did you hate both of them?")
                fields_to_csv["why_user_pref"] = why_user_pref
                print("why_user_pref", why_user_pref)
            if st.button("Submit"):
                if st.session_state.email == "":
                    send_email(fields_to_csv)
                    st.write("Thank you for completing this questionnaire :)")
