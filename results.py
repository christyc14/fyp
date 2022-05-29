import json
import numpy as np
from pydantic import BaseModel
from pydantic.dataclasses import dataclass

# from dataclasses_json import dataclass_json
from typing import Dict, List, Optional

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


class SkinInfo(BaseModel):
    skin_tone: str
    skin_type: str
    skin_concern: str


class ML_input(BaseModel):
    categories: List[str]
    skin_info: SkinInfo


class FollowUpQuestionData(BaseModel):
    tried_products: List[str]
    willing: bool


class SephoraInfo(BaseModel):
    sephora_username: Optional[str]
    user_has_sephora: bool
    sephora_not_in_list: Optional[bool]


class NPSScore(BaseModel):
    score: int
    comment: str


class ComparisonAnswers(BaseModel):
    user_pref: str
    why_user_pref: Optional[List[str]]
    why_user_pref_other: Optional[str]
    user_consistent: Optional[str]
    user_when: Optional[str]


class FormData(BaseModel):
    top_3: Dict[str, List[str]]
    recommendations: Dict[Stage, Dict[Category, List[str]]]
    sephora_info: SephoraInfo
    ml_input: ML_input
    follow_up_questions: Dict[Stage, Dict[Category, FollowUpQuestionData]]
    nps_scores: Dict[Stage, NPSScore]
    comparison_answers: Optional[ComparisonAnswers] = None


data: List[FormData] = []

with open("data.jsonlines") as f:
    for line in f:
        data.append(FormData(**json.loads(line)))

ml_user_pref = "Skin Type Recommender"
svd_user_pref = "Top 3 Products Recommender"

for stage in ["ML", "SVD"]:
    recommendations_tried_per_user = np.mean(
        [
            np.mean(
                [
                    len(follow_up_data.tried_products)
                    for _, follow_up_data in user.follow_up_questions[stage].items()
                ]
            )
            for user in data
        ]
    )

    print(
        stage,
        "Average number of recommendations tried per user, normalized: ",
        recommendations_tried_per_user,
    )

    number_willing = np.mean(
        [
            np.mean(
                [
                    int(follow_up_data.willing)
                    for _, follow_up_data in user.follow_up_questions[stage].items()
                ]
            )
            for user in data
        ]
    )

    print(stage, number_willing)

    num_detractors = len([0 for user in data if user.nps_scores[stage].score <= 6])
    num_promoters = len([0 for user in data if user.nps_scores[stage].score > 8])
    nps = (num_promoters - num_detractors) / len(data)
    print(stage, "NPS: ", nps)

    print(stage, "why user is a detractor:")
    quali_why_detract = print(
        *[
            user.nps_scores[stage].comment
            for user in data
            if user.nps_scores[stage].score <= 6
        ],
        sep="\n"
    )
    print(stage, "why user is a promoter:")
    quali_why_promote = print(
        *[
            user.nps_scores[stage].comment
            for user in data
            if user.nps_scores[stage].score > 8
        ],
        sep="\n"
    )
    print(stage, "why user is a passive:")
    quali_why_passive = print(
        *[
            user.nps_scores[stage].comment
            for user in data
            if user.nps_scores[stage].score >6 and user.nps_scores[stage].score <= 8
        ],
        sep="\n"
    )

why_user_pref_svd_personal = len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == svd_user_pref
        and "Personalization" in user.comparison_answers.why_user_pref
    ]
) / len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == svd_user_pref
    ]
)
why_user_pref_svd_ease = len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == svd_user_pref
        and "Ease of use" in user.comparison_answers.why_user_pref
    ]
) / len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == svd_user_pref
    ]
)
print("Reason for preferring SVD: \n")
why_user_pref_svd_comment = print(
    [
        user.comparison_answers.why_user_pref_other
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == svd_user_pref
    ]
)

num_user_pref_ml_personal = len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == ml_user_pref
        and "Personalization" in user.comparison_answers.why_user_pref
    ]
) / len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == ml_user_pref
    ]
)
why_user_pref_ml_ease = len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == ml_user_pref
        and "Ease of use" in user.comparison_answers.why_user_pref
    ]
) / len(
    [
        0
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == ml_user_pref
    ]
)
print("Reason for preferring ML:\n")
why_user_pref_ml_comment = print(
    *[
        user.comparison_answers.why_user_pref_other
        for user in data
        if user.comparison_answers != None
        and user.comparison_answers.user_pref == ml_user_pref
    ],
    sep="\n"
)


percent_users_prefer_ml = (
    len(
        [
            0
            for user in data
            if user.comparison_answers != None
            and user.comparison_answers.user_pref == "Skin Type Recommender"
        ]
    )
    / len([
            0
            for user in data
            if user.comparison_answers != None])
    * 100
)
percent_users_prefer_svd = (
    len(
        [
            0
            for user in data
            if user.comparison_answers != None
            and user.comparison_answers.user_pref == "Top 3 Products Recommender"
        ]
    )
    / len([
            0
            for user in data
            if user.comparison_answers != None])
    * 100
)
percent_users_consistent = (
    len(
        [
            0
            for user in data
            if user.comparison_answers != None
            and user.comparison_answers.user_consistent == "Yes"
        ]
    )
    / len([
            0
            for user in data
            if user.comparison_answers != None])
    * 100
)

print("Percent users prefer ML: ", percent_users_prefer_ml)
print("Percent users prefer SVD: ", percent_users_prefer_svd)
print("Percent users are consistent: ", percent_users_consistent)
print("When they would use either consistently: \n")
user_when = print(
    *[
        user.comparison_answers.user_when
        for user in data
        if user.comparison_answers != None
    ],
    sep="\n"
)
