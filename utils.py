from __future__ import print_function
import pandas as pd

import streamlit as st
from typing import List, Tuple


def make_clickable(link: str) -> str:
    # target _blank to open new window
    # extract clickable text to display for your link
    return f'<a target="_blank" href="{link}">Sephora US Link</a>'


def build_ml_predictor_input(df: pd.DataFrame):
    ml_df = pd.DataFrame()
    ml_df["ingredients"] = df["ingredients"]
    ml_df["category"] = df["category"]
    ml_df["skin_tone"] = st.session_state.form_data.ml_input.skin_info.skin_tone
    ml_df["skin_type"] = st.session_state.form_data.ml_input.skin_info.skin_type
    ml_df["skin_concern"] = st.session_state.form_data.ml_input.skin_info.skin_concern
    ml_df["product_name"] = df["product_name"]
    ml_df["url"] = df["url"]

    return ml_df


@st.experimental_memo
def get_users_with_multiple_reviews(df: pd.DataFrame) -> List[str]:
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


def display_recommendations(df: pd.DataFrame) -> None:
    disp = df[["product_name", "url"]]
    disp["url"] = disp["url"].apply(make_clickable)
    disp = disp[["product_name", "url"]].to_html(escape=False)
    st.write(disp, unsafe_allow_html=True)
