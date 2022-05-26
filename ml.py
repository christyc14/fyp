from dataclasses import dataclass
from dataclasses_json import dataclass_json
import streamlit as st
import pandas as pd
from typing import Dict, List
from loaders import load_data, load_model
from svd import content_recommender
from svd import collab_recommender
from autogluon.tabular import TabularPredictor
from utils import (
    build_ml_predictor_input,
    display_recommendations,
    get_users_with_multiple_reviews,
)

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

@dataclass_json
@dataclass
class FormData:
    recommendations: Dict[Stage, Dict[Category, List[str]]]
    ml_input: ML_input

df: pd.DataFrame = load_data()
predictor: TabularPredictor = load_model()
product_categories: List[str] = list(set(df["category"].values))

if "form_data" not in st.session_state:
    st.session_state.form_data = FormData(
        {"ML": {}},
        ML_input([], SkinInfo(None, None, None)),
    )

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
for category in st.session_state.form_data.ml_input.categories:
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

st.success("Thanks for using this product!")