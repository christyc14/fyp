from __future__ import print_function
from dataclasses import dataclass
import json
from dataclasses_json import dataclass_json

import streamlit as st
import os
import pandas as pd
from typing import Dict, List
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from pprint import pprint
from loaders import load_data, load_model
from svd import content_recommender
from svd import collab_recommender
from autogluon.tabular import TabularPredictor
import random

from utils import (
    build_ml_predictor_input,
    display_recommendations,
    get_users_with_multiple_reviews,
    pad_selected_products,
)

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

Stage = str
Category = str


@dataclass
class SkinInfo:
    skin_tone: str
    skin_type: str
    skin_concern: str


@dataclass
class ML_input:
    categories: List[str]
    skin_info: SkinInfo


@dataclass
class FollowUpQuestionData:
    tried_products: List[str]
    willing: bool


@dataclass
class SephoraInfo:
    sephora_username: str
    user_has_sephora: bool
    sephora_not_in_list: bool


@dataclass
class NPSScore:
    score: int
    comment: str


@dataclass_json
@dataclass
class FormData:
    top_3: Dict[str, List[str]]
    recommendations: Dict[Stage, Dict[Category, List[str]]]
    sephora_info: SephoraInfo
    ml_input: ML_input
    follow_up_questions: Dict[Stage, Dict[Category, FollowUpQuestionData]]
    nps_scores: Dict[Stage, NPSScore]


df: pd.DataFrame = load_data()
predictor: TabularPredictor = load_model()
product_categories: List[str] = list(set(df["category"].values))

if "form_data" not in st.session_state:
    st.session_state.form_data = FormData(
        {},
        {"ML": {}, "SVD": {}},
        SephoraInfo(None, None, None),
        ML_input([], SkinInfo(None, None, None)),
        {"ML": {}, "SVD": {}},
        {},
    )


def follow_up_questions(stage: Stage, df: pd.DataFrame, category: Category):
    st.write("Have you already tried any of these products? If so, please select them.")
    tried_products = [
        st.checkbox(product_name, key=f"{stage}_{category}_tried_products")
        for product_name in df["product_name"].values
    ]
    willing = st.radio(
        "Would you be willing to try any of these products?",
        key=f"{stage}_{category}_willing",
        options=["Yes", "No"],
        index=1,
    )
    st.session_state.form_data.follow_up_questions[stage][
        category
    ] = FollowUpQuestionData(df["product_name"][tried_products].tolist(), willing)


def category_selection(_):
    st.write("Which categories would you like recommendations for?")
    checkboxes = [st.checkbox(cat, value=False) for cat in product_categories]
    selected_categories = [
        cat for i, cat in enumerate(product_categories) if checkboxes[i]
    ]
    st.session_state.form_data.ml_input.categories = selected_categories
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


def require_preference_reason(why_user_pref, why_user_pref_other) -> bool:
    print("holo")
    print(why_user_pref, why_user_pref_other)
    if why_user_pref == ["Other"] and why_user_pref_other == "":
        st.error("Please enter a reason")
        return False
    if why_user_pref == []:
        st.error("Please select a reason")
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
            follow_up_questions("SVD", recommendations, cat)
            if st.form_submit_button():
                st.session_state.svd_cat_index += 1
                placeholder.empty()
            else:
                st.stop()

    if not st.session_state.svd_complete:
        with st.form("nps_svd"):
            score = st.slider(
                "How likely is it would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
                min_value=0,
                max_value=10,
                value=1,
                key="reccommend",
            )
            comment = st.text_input(
                "Please explain why you gave that score.", key="nps_svd_reason"
            )
            if st.form_submit_button():
                st.session_state.form_data.nps_scores["SVD"] = NPSScore(score, comment)
                st.session_state.svd_complete = True
                st.success("Thanks for your feedback!")
                return True
    else:
        return True


if "ml_stage_counter" not in st.session_state:
    st.session_state.ml_stage_counter = 0

if "ml_stage_input" not in st.session_state:
    st.session_state.ml_stage_input = None


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
    st.session_state.form_data.ml_input.skin_info = SkinInfo(
        skin_tone, skin_type, skin_concern
    )


ML_stages = [(category_selection, validate_category_selection), (skin_info, None)]


def ml():
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
    while st.session_state.ml_cat_index < len(
        st.session_state.form_data.ml_input.categories
    ):
        category = st.session_state.form_data.ml_input.categories[
            st.session_state.ml_cat_index
        ]
        placeholder = st.empty()
        with placeholder.form(key=f"ml_recommendation_{category}"):
            ml_df = build_ml_predictor_input(df)
            ml_df["pred_rating"] = predictor.predict(
                ml_df.drop(["product_name", "url"], axis=1)
            )
            ml_df = ml_df.sort_values(by="pred_rating", ascending=False)
            ml_df = ml_df[ml_df["category"] == category]

            recommendations: pd.DataFrame = ml_df.head(5)
            st.write(f"Here are your {category} recommendations:")
            display_recommendations(recommendations)
            st.session_state.form_data.recommendations["ML"][
                category
            ] = recommendations["product_name"].tolist()
            follow_up_questions("ML", recommendations, category)
            if st.form_submit_button():
                st.session_state.ml_cat_index += 1
                placeholder.empty()
            else:
                st.stop()
    if not st.session_state.ml_complete:
        with st.form("nps_ml"):
            score = st.slider(
                "How likely is it would you recommend this recommender to someone else? (1 = not at all likely, 10 = extremely likely)",
                min_value=0,
                max_value=10,
                value=1,
                key="ml_reccommend",
            )
            comment = st.text_input(
                "Please explain why you gave that score.", key="ml_nps_reason"
            )
            if st.form_submit_button():
                st.session_state.form_data.nps_scores["ML"] = NPSScore(score, comment)
                st.session_state.ml_complete = True
                st.success("Thanks for your feedback!")
                return True
    else:
        return True


if "k" not in st.session_state:
    st.session_state["k"] = random.random()

if "email" not in st.session_state:
    st.session_state["email"] = ""

if "svd_complete" not in st.session_state:
    st.session_state["svd_complete"] = False

if "ml_complete" not in st.session_state:
    st.session_state["ml_complete"] = False

st.title("FYP Questionnaire")
st.caption(
    "Please answer the following questions. There are three parts, one for Recommender A, and one for Recommender B, and one to compare the two."
)
if st.session_state.k < 0.5:
    if svd() is not None:
        st.write("Thank you for completing part 1 :)")
        st.write("Part 2 is a different recommender.")
        ml()

else:
    if ml() is not None:
        st.write("Thank you for completing part 1 :)")
        st.write("Part 2 is a different recommender.")
        svd()

if st.session_state["ml_complete"] and st.session_state["svd_complete"]:
    with st.form("part3"):
        st.write("Thank you for completing part 2 :)")
        st.write("Part 3 contains questions comparing part 1 and part 2.")
        user_pref = st.radio(
            "Which of the two did you prefer?",
            options=["Top 3 Products Recommender", "Skin Type Recommender"],
        )
        why_user_pref = st.multiselect(
            "Why did you prefer that recommender?",
            options=["", "Personalization", "Ease of use", "Other"],
        )
        why_user_pref_other = st.text_input(
            "If you selected other, please explain why."
        )
        user_consistent = st.radio(
            "Would you use either of them consistently?", options=["Yes", "No"]
        )
        if user_consistent == "Yes":
            user_when = st.text_input("When would you use it?")
        else:
            user_not_use = st.text_input("Why not?")
        if st.form_submit_button("Submit") and require_preference_reason(
            why_user_pref, why_user_pref_other
        ):
            with open("data.jsonlines", "a") as f:
                f.write(st.session_state.form_data.to_json())
            print(st.session_state.form_data.to_json())
            st.success("Thank you for completing this questionnaire :)")
        else:
            st.stop()
