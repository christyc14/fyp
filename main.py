from calendar import c
from typing import Dict, List, Union
from zlib import DEF_BUF_SIZE
import json_lines
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import json
from scipy.sparse.linalg import svds
from scipy.spatial import distance

if __name__ == "__main__":
    print("Hello world!")

products: List[Dict[str, Union[str, List[str]]]] = []
# input data into List
with open("cbscraper/product_urls_with_reviews.jsonlines", "rb") as f:
    unique = set()
    lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ["json_element"]
    df_inter["json_element"].apply(json.loads)
    df = pd.json_normalize(df_inter["json_element"].apply(json.loads))

# to save myself if i do something dumb and run the scraper without deleting the .jsonlines file
df.drop_duplicates(subset=["url"], inplace=True)

# option: category of product, eg cleanser
categories = set(df.category.values)
# filter data by given option


def preprocess_ingredients(ingredients):
    processed_ingredients = []
    for i in range(len(ingredients)):
        processed_ingredient = re.sub(
            r"\(([^)]*)\)|(([0-9]\d{0,2}(\.\d{1,3})*(,\d+)?)(%|mg|units))|(<\/?i>)|(\/.+)|(\\.+)|\[([^\]]*)\]",
            "",
            ingredients[i],
        ).strip()
        if (
            processed_ingredient.lower() == "water"
            or processed_ingredient.lower() == "aqua"
            or processed_ingredient.lower() == "eau"
        ):
            processed_ingredient = "Water"
        processed_ingredients.append(processed_ingredient)
    return processed_ingredients


def content_recommender(opt, _item1, _item2, _item3, num_recs, df):
    df = df[df.category == opt]
    df["ingredients"] = df["ingredients"].map(preprocess_ingredients)
    mlb = MultiLabelBinarizer()
    output = mlb.fit_transform(df.ingredients.values)
    df = df.drop(["ingredients"], axis=1)
    # tsne to reduce dimensionality
    model = TSNE(n_components=2, learning_rate=200)
    tsne_features = model.fit_transform(output)

    df["X"] = tsne_features[:, 0]
    df["Y"] = tsne_features[:, 1]
    df["dist"] = 0.0
    item1 = df[df["product_name"] == _item1]
    item2 = df[df["product_name"] == _item2]
    item3 = df[df["product_name"] == _item3]

    p1 = np.array([item1["X"], item1["Y"]]).reshape(1, -1)
    p2 = np.array([item2["X"], item2["Y"]]).reshape(1, -1)
    p3 = np.array([item3["X"], item3["Y"]]).reshape(1, -1)
    for ind, item in df.iterrows():
        pn = np.array([item.X, item.Y]).reshape(-1, 1)
        df.at[ind, "dist"] = min(
            distance.chebyshev(p1, pn),
            distance.chebyshev(p2, pn),
            distance.chebyshev(p3, pn),
        )
    df = df.sort_values("dist")

    mask = df[
        (df["product_name"].ne(item1["product_name"]))
        & (df["product_name"].ne(item2["product_name"]))
        & (df["product_name"].ne(item3["product_name"]))
    ]
    return mask.head(num_recs)


df_tmp = content_recommender(
    "TONER",
    "Squalane + BHA Pore-Minimizing Toner",
    "Mandelic Acid + Superfood Unity Exfoliant",
    "Watermelon Glow PHA +BHA Pore-Tight Toner",
    10,
    df,
)
# print(df_tmp[["brand", "product_name", "url", "avg_rating"]].head(10))


def collab_recommender(df_tmp, num_recs, username):
    reviews = df_tmp.explode("review_data")
    dictionary = {}
    reviews["username"] = reviews["review_data"].apply(lambda x: x["UserNickname"])
    reviews["rating"] = reviews["review_data"].apply(lambda x: x["Rating"])
    grouped_reviews = reviews.groupby("username")["review_data"].apply(list)
    multiple_rating_users = set(grouped_reviews[grouped_reviews.map(len) > 1].index)
    multi_reviews = reviews[reviews.username.isin(multiple_rating_users)]
    products_reviewed_per_user = {u: set() for u in multiple_rating_users}
    product_index = dict(zip(df_tmp["url"].values, range(len(df_tmp["url"]))))
    username_index = dict(zip(multiple_rating_users, range(len(multiple_rating_users))))
    matrix = np.zeros((len(multiple_rating_users), len(df_tmp["url"])))
    for user, rating, url in zip(
        multi_reviews.username.values,
        multi_reviews.rating.values,
        multi_reviews.url.values,
    ):
        matrix[username_index[user]][product_index[url]] = rating
        products_reviewed_per_user[user].add(url)

    ss = StandardScaler()
    normatrix = ss.fit_transform(matrix)
    U, S, V = svds(normatrix)
    all_user_predicted_rating = ss.inverse_transform(U @ np.diag(S) @ V)

    preds_df = pd.DataFrame(
        all_user_predicted_rating, columns=product_index, index=username_index
    )

    sorted_user_preds = preds_df.loc[username].sort_values(ascending=False)
    sorted_user_preds = sorted_user_preds[~sorted_user_preds.index.isin(products_reviewed_per_user[username])]
    sorted_user_preds = sorted_user_preds.head(num_recs)
    # we want those that they haven't already tested
    collab_df = pd.merge(
        df_tmp,
        sorted_user_preds.to_frame(),
        left_on="url",
        right_index=True,
        how="right",
    )
    collab_df.rename(columns={username: "pred_rating"}, inplace=True)
    return collab_df


print(
    collab_recommender(df_tmp, 10, "klainee")[
        ["brand", "product_name", "url", "pred_rating"]
    ]
)
