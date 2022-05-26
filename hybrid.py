import streamlit as st
import pandas as pd
from loaders import load_data
from typing import Dict, List
from utils import (
    display_recommendations,
    get_users_with_multiple_reviews,
    pad_selected_products,
)
from dataclasses_json import dataclass_json
from dataclasses import dataclass
from svd import content_recommender, collab_recommender

Stage = str
Category = str


df: pd.DataFrame = load_data()
product_categories: List[str] = list(set(df["category"].values))

@dataclass
class SephoraInfo:
    sephora_username: str
    user_has_sephora: bool
    sephora_not_in_list: bool

@dataclass_json
@dataclass
class FormData:
    top_3: Dict[str, List[str]]
    recommendations: Dict[Stage, Dict[Category, List[str]]]
    sephora_info: SephoraInfo

def category_selection(_):
    st.write("Which categories would you like recommendations for?")
    checkboxes = [st.checkbox(cat, value=False) for cat in product_categories]
    selected_categories = [
        cat for i, cat in enumerate(product_categories) if checkboxes[i]
    ]
    return selected_categories

def validate_category_selection(selected_categories):
    if len(selected_categories) == 0:
        st.error("Please select at least one category")
        return False
    return True


def top3_product_selection(selected_categories):
    top_3: Dict[str, List[str]] = {}
    st.write("What are your top 3 products for each category?")
    for category in selected_categories:
        top_3[category] = st.multiselect(
            category,
            df[df.category == category]["product_name"].values,
        )
    st.session_state.form_data.top_3 = top_3
    return top_3


def validate_top3_product_selection(top_3) -> bool:
    if any([len(v) > 3 for v in top_3.values()]):
        st.error("Please select at most 3 products for each category")
        return False
    elif any([len(v) == 0 for v in top_3.values()]):
        st.error("Please select at least 1 product for each category")
        return False
    return True

def user_has_sephora(_):
    st.session_state.form_data.sephora_info.user_has_sephora = (
        st.radio(
            "Do you have a Sephora (US) account?",
            options=["Yes", "No"],
            index=1,
        )
        == "Yes"
    )
    return st.session_state.form_data.sephora_info.user_has_sephora


def sephora_info(_):
    st.session_state.form_data.sephora_info.sephora_username = st.selectbox(
        "Please select your username from dropdown",
        sorted(get_users_with_multiple_reviews(df)),
    )
    st.session_state.form_data.sephora_info.sephora_not_in_list = st.checkbox(
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

if "selected_categories" not in st.session_state:
    st.session_state.selected_categories = []

if "form_data" not in st.session_state:
    st.session_state.form_data = FormData(
        {},
        {"SVD": {}},
        SephoraInfo(None, None, None),
    )

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
for cat, products in list(st.session_state.top3.items()):
    padded_products = pad_selected_products(products)
    if (
        st.session_state.form_data.sephora_info.user_has_sephora
        and not st.session_state.form_data.sephora_info.sephora_not_in_list
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
        print("cat = ", cat)
        print("products = ", padded_products)
        print("df = ", df)
        recommendations = content_recommender(
            cat,
            *(padded_products),
            df,
        )

    recommendations = recommendations.head(5)
    print("recommendations", recommendations)
    st.write(f"Here are your {cat} recommendations:")
    display_recommendations(recommendations)
    st.session_state.form_data.recommendations["SVD"][cat] = recommendations[
        "product_name"
    ].tolist()

st.success("Thanks for using this product!")