from functools import partial
import streamlit as st
import os
import pandas as pd
import numpy as np
from calendar import c
from typing import Dict, List, Union
from zlib import DEF_BUF_SIZE
import json_lines
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import json
from scipy.sparse.linalg import svds
from scipy.spatial import distance
from autogluon.tabular import TabularPredictor

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


df: pd.DataFrame = load_data()
# products: List[Dict[str, Union[str, List[str]]]] = []
product_categories: List[str] = list(set(df.category.values))

def svd(df, product_categories):
    st.write("1. Which categories would you like recommendations for?")
    checkboxes = [st.checkbox(cat, value=False) for cat in product_categories]
    top_3: Dict[str, List[str]] = {}
    if any(checkboxes):
        st.write("2. Which are your top 3 products for each category?")
        for i, checkbox in enumerate(checkboxes):
            if checkbox:
                top_3[product_categories[i]] = st.multiselect(
                    product_categories[i],
                    df[df.category == product_categories[i]]["product_name"].values,
                )
        if any([len(v) > 3 for v in top_3.values()]):
            st.warning("Please select at most 3 products for each category")
        else:
            user_has_sephora: bool = (
                st.radio("3. Do you have a Sephora (US) account?", options=["Yes", "No"], index=1)
                == "Yes"
            )
            if user_has_sephora:
                multiple_reviews: bool = (
                    st.radio(
                        "3a. Have you written 2 or more reviews?", options=["Yes", "No"]
                    )
                    == "Yes"
                )
            if user_has_sephora and multiple_reviews:
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
                    reviews = cbf.explode("review_data")
                    reviews["username"] = reviews["review_data"].apply(
                        lambda x: x["UserNickname"]
                    )
                    reviews["rating"] = reviews["review_data"].apply(lambda x: x["Rating"])
                    grouped_reviews = reviews.groupby("username")["review_data"].apply(list)
                    multiple_rating_users = [""] + list(
                        set(grouped_reviews[grouped_reviews.map(len) > 1].index)
                    )
                    username = st.selectbox(
                        "3b. Please select your username from dropdown",
                        sorted(multiple_rating_users),
                    )
                    user_not_in_list = st.checkbox(
                        "Please check here if your username is not in this list"
                    )
                    if user_not_in_list:
                        user_has_sephora = False
                    elif username != "":
                        st.write(f"{cat}")
                        cf = collab_recommender(cbf, 5, username)
                        cf["url"] = cf["url"].apply(make_clickable)
                        cf = cf[["brand", "product_name", "url", "pred_rating"]].to_html(
                            escape=False
                        )
                        st.write(cf, unsafe_allow_html=True)

            if not user_has_sephora:  # if no sephora acc
                st.write("Here are your recommendations:")
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
                        5,
                        df,
                    )
                    # link is the column with hyperlinks
                    cbf_disp = cbf[["brand", "product_name", "url"]]
                    cbf_disp["url"] = cbf_disp["url"].apply(make_clickable)
                    cbf_disp = cbf_disp[["brand", "product_name", "url"]].to_html(
                        escape=False
                    )
                    st.write(cbf_disp, unsafe_allow_html=True)
                    have_tried = st.radio(
                        "4. Have you tried any of these products?",
                        options=["Yes", "No"],
                        key=f"{cat}_tried",
                    )
                    if have_tried == "Yes":
                        st.write("5. Which of these have you tried?")
                        tried_products = [
                            st.checkbox(item.product_name) for _, item in cbf.iterrows()
                        ]
                    willing = st.radio(
                        "6. Would you be willing to try any of these products?",
                        key=f"{cat}_willing",
                        options=["Yes", "No"],
                    )
                    if willing == "Yes":
                        willing_products_temp = []
                        for _, item in cbf.iterrows():
                            if item.product_name not in tried_products:
                                willing_products_temp.append(item.product_name)
                        willing_products = [
                            st.checkbox(item, key=f"_{item}")
                            for item in willing_products_temp
                        ]

                reccommend = st.slider(
                    "7. How likely would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
                    min_value=0,
                    max_value=10,
                    value=1,
                )
                nps_reason = st.text_input("8. Please explain why you gave that score.")
                st.write("Thank you for completing part 1 :)")

def ml(df, product_categories):
    st.write(
        "Part 2 is a different recommender."
    )
    ml_pred = TabularPredictor.load("src/agModels-predictClass/")
    #ask questions
    st.write("1. Which categories would you like recommendations for?")
    checkboxes_ml = [st.checkbox(cat, value=False, key = f"{cat}_ml") for cat in product_categories]
    cat_ml = [cat for i, cat in enumerate(product_categories) if checkboxes_ml[i]]
    print(cat_ml)
    if any(checkboxes_ml):
        skin_tone = st.selectbox("2. What is your skin tone?", options=["", "porcelain", "fair", "light", "medium", "tan", "olive", "deep", "dark", "ebony"])
        if skin_tone != "":
            skin_type = st.selectbox("3. What is your skin type?", options=["", "normal", "combination", "dry", "oily"])
            if skin_type != "":
                skin_concern = st.selectbox("4. What are your most important skincare concern?", options = ["acne", "aging", "blackheads", "darkCircles", "dullness", "sensitivity", "redness", "sunDamage", "cellulite", "pores", "unevenSkinTones", "cuticles", "calluses", "stretchMarks", "puffiness"])
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
    for cat in cat_ml:
        df_tmp = df_tmp[df_tmp["category"] == cat]
        ml_disp = df_tmp[["brand", "product_name", "url"]].head(5)
        ml_disp["url"] = ml_disp["url"].apply(make_clickable)
        ml_disp = ml_disp[["brand", "product_name", "url"]].to_html(
            escape=False
        )
        st.write(ml_disp, unsafe_allow_html=True)
        have_tried_ml = st.radio(
                "5. Have you tried any of these products?",
                options=["Yes", "No"],
                key=f"{cat}_tried",
            )
        if have_tried_ml == "Yes":
            st.write("6. Which of these have you tried?")
            tried_products_ml = [
                st.checkbox(item.product_name) for _, item in df_tmp.iterrows()
            ]
        willing_ml = st.radio(
            "7. Would you be willing to try any of these products?",
            key=f"{cat}_willing",
            options=["Yes", "No"],
        )
        if willing_ml == "Yes":
            willing_products_temp_ml = []
            for _, item in ml_disp.iterrows():
                if item.product_name not in tried_products_ml:
                    willing_products_temp.append(item.product_name)
            willing_products_ml = [
                st.checkbox(item, key=f"_{item}")
                for item in willing_products_temp
            ]

    reccommend_ml = st.slider(
        "7. How likely would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
        min_value=0,
        max_value=10,
        value=1,
        key = "ml_reccommend",
    )
    nps_reason_ml = st.text_input("8. Please explain why you gave that score.", key = "ml_nps_reason")
    st.write("Thank you for completing part 2 :)")


st.title("FYP Questionnaire")
st.caption("Please answer the following questions. There are three parts, one for Recommender A, and one for Recommender B, and one to compare the two.")

st.write("Part 3 contains questions comparing part 1 and part 2.")
user_pref = st.radio("Which of the two did you prefer?", options=["Part 1", "Part 2", "Hated both :("])
if user_pref == "Part 1" or "Part 2":
    why_user_pref = st.text_input(f"Why did you prefer {user_pref}?")
    user_consistent = st.radio("Would you use either of them consistently?", options=["Yes", "No"])
    if user_consistent == "Yes":
        user_when = st.text_input("When would you use it?")
    else:
        user_not_use = st.text_input("Why not?")
else:
    why_user_pref = st.text_input("Why did you hate both of them?")
st.write("Thank you for completing this questionnaire :)")