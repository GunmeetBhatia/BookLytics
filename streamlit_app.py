import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
import sklearn
from packaging import version

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(page_title="BookLytics", layout="wide")

# ======================================================
# SIMPLE CLEAN THEME
# ======================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(-45deg, #0B0C10, #1F2833, #0B0C10);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    color: #C5C6C7;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

[data-testid="stSidebar"] {
    background-color: #0B0C10;
    border-right: 1px solid #1F2833;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 700;
    color: #66FCF1;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #C5C6C7;
    margin-bottom: 40px;
}

.card {
    background: rgba(31, 40, 51, 0.7);
    backdrop-filter: blur(10px);
    padding: 25px;
    border-radius: 16px;
    border: 1px solid rgba(102,252,241,0.2);
    margin-bottom: 30px;
}

h2, h3 {
    color: #45A29E;
}

.stButton>button {
    background: linear-gradient(90deg, #45A29E, #66FCF1);
    color: #0B0C10;
    border-radius: 8px;
    border: none;
    padding: 0.6em 1.4em;
    font-weight: 600;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102,252,241,0.3);
}

[data-testid="metric-container"] {
    background: #1F2833;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #45A29E30;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>BookLytics</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Book Intelligence & Market Analytics Platform</div>", unsafe_allow_html=True)

# ======================================================
# PATHS
# ======================================================

DATA_PATH = Path("Books_df.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CLS_MODEL_PATH = MODEL_DIR / "classifier.joblib"
REG_MODEL_PATH = MODEL_DIR / "regressor.joblib"

# ======================================================
# HELPERS
# ======================================================

def clean_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    return float(m.group(1)) if m else np.nan


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        return pd.DataFrame()

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "price" in df.columns:
        df["price_clean"] = df["price"].apply(clean_price)
    else:
        df["price_clean"] = 0

    for col in ["rating", "avg_rating", "average_rating"]:
        if col in df.columns:
            df["rating_clean"] = pd.to_numeric(df[col], errors="coerce")
            break

    df["author"] = df.get("author", "Unknown").fillna("Unknown")
    df["title"] = df.get("title", "").fillna("")
    df["num_ratings"] = df.get("num_ratings", 0)

    if "rating_clean" in df.columns:
        df["target"] = (df["rating_clean"] >= 4.0).astype(int)

    return df


def build_preprocessor():
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=800, stop_words="english")),
        ("svd", TruncatedSVD(n_components=15, random_state=42))
    ])

    return ColumnTransformer([
        ("title", text_pipe, "title"),
        ("author", ohe, ["author"]),
        ("num", StandardScaler(), ["price_clean", "num_ratings"])
    ])


def train_models(df):
    X = df[["title", "author", "price_clean", "num_ratings"]].fillna(0)
    y_cls = df["target"]
    y_reg = df["rating_clean"].fillna(df["rating_clean"].median())

    clf = Pipeline([
        ("pre", build_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=42))
    ])

    reg = Pipeline([
        ("pre", build_preprocessor()),
        ("model", RandomForestRegressor(n_estimators=80, n_jobs=-1, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, stratify=y_cls, random_state=42
    )

    clf.fit(X_train, y_train)
    reg.fit(X, y_reg)

    

    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)


# ======================================================
# LOAD DATA
# ======================================================

uploaded = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
df = load_data(uploaded)

if df.empty:
    st.warning("Upload dataset to continue.")
    st.stop()

    train_models(df)

    clf, reg = train_models(df)

# ======================================================
# DATA OVERVIEW
# ======================================================

st.header("Dataset Overview")
st.write("Shape:", df.shape)
# FULL DATASET VIEW
st.dataframe(df, use_container_width=True)
#  Search
search = st.text_input("Search Book")
filtered_titles = df[
    df["title"].str.contains(search, case=False, na=False)
]["title"]
#  Select
selected_book = st.selectbox(
    "Select Book",
    filtered_titles.unique()
)
#  Show selected book
if selected_book:
    book_data = df[df["title"] == selected_book]
    st.write(book_data)

# ======================================================
# PREDICTION (TITLE + AUTHOR ONLY)
# ======================================================

st.header("Book Rating Prediction")

title = st.text_input("Title", key="pred_title")
author = st.text_input("Author", key="pred_author")

if st.button("Predict", key="predict_btn"):


        clf = joblib.load(CLS_MODEL_PATH)
        reg = joblib.load(REG_MODEL_PATH)

        # Auto-fill missing numeric fields
        Xnew = pd.DataFrame([{
            "title": title,
            "author": author,
            "price_clean": 0,
            "num_ratings": 0
        }])

        prob = clf.predict_proba(Xnew)[0, 1]
        rating = reg.predict(Xnew)[0]

        st.metric("Probability ≥ 4.0", f"{prob:.2f}")
        st.metric("Predicted Rating", f"{rating:.2f}")

    
# ======================================================
# MARKET SEGMENTATION (RESTORED)
# ======================================================

# ======================================================
# MARKET SEGMENTATION (DETAILED VERSION)
# ======================================================

st.header("Market Segmentation")

st.markdown("""
Books are grouped into natural market segments based on:
- Title similarity
- Author patterns
- Pricing structure
- Popularity (rating counts)
""")

k = st.slider("Number of Clusters", 2, 6, 4)

if st.button("Generate Segments", key="cluster_btn"):

    pre = build_preprocessor()
    X = df[["title", "author", "price_clean", "num_ratings"]].fillna(0)
    M = pre.fit_transform(X)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(M)



    # --------------------------------------------------
    # Segment Distribution
    # --------------------------------------------------

    st.subheader("Segment Distribution")
    st.bar_chart(df["cluster"].value_counts().sort_index())

    # --------------------------------------------------
    # Segment Profile (Numerical Summary)
    # --------------------------------------------------

    if "rating_clean" in df.columns:

        st.subheader("Segment Characteristics")

        profile = df.groupby("cluster").agg({
            "price_clean": "mean",
            "rating_clean": "mean",
            "num_ratings": "mean"
        }).round(2)

        profile.columns = [
            "Avg Price",
            "Avg Rating",
            "Avg Popularity"
        ]

        st.dataframe(profile)

        # --------------------------------------------------
        # Automatic Interpretation
        # --------------------------------------------------

        st.subheader("Segment Interpretation")

        for cluster_id in profile.index:

            avg_price = profile.loc[cluster_id, "Avg Price"]
            avg_rating = profile.loc[cluster_id, "Avg Rating"]
            avg_pop = profile.loc[cluster_id, "Avg Popularity"]

            description = f"Cluster {cluster_id}: "

            if avg_price > df["price_clean"].mean():
                description += "Premium priced, "
            else:
                description += "Budget friendly, "

            if avg_rating > df["rating_clean"].mean():
                description += "well rated, "
            else:
                description += "moderately rated, "

            if avg_pop > df["num_ratings"].mean():
                description += "highly popular segment."
            else:
                description += "niche audience segment."

            st.write(description)

    # --------------------------------------------------
    # Sample Books from Each Cluster
    # --------------------------------------------------

    st.subheader("Sample Books Per Segment")

    sample_books = df.groupby("cluster").head(5)[
        ["cluster", "title", "author"]
    ].reset_index(drop=True)

    st.dataframe(sample_books)
# ======================================================
# RECOMMENDATIONS (FIXED INPUT)
# ======================================================

st.header("Book Recommendations")

@st.cache_resource
def build_recommender(data):
    pre = build_preprocessor()
    M = pre.fit_transform(
        data[["title", "author", "price_clean", "num_ratings"]].fillna(0)
    )
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    nn.fit(M)
    return nn, M

nn, M = build_recommender(df)

query = st.text_input("Enter exact or partial book title", key="rec_query")

if st.button("Recommend", key="rec_btn"):
    matches = df[df["title"].str.contains(query, case=False, na=False)]

    if matches.empty:
        st.warning("Book not found.")
    else:
        idx = matches.index[0]
        _, indices = nn.kneighbors(M[idx].reshape(1, -1))
        st.dataframe(
            df.iloc[indices[0]][["title", "author"]].reset_index(drop=True)
        )