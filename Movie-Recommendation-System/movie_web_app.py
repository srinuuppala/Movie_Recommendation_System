# streamlit_app.py
"""
Optimized Streamlit Movie Recommender (content-based) with lazy poster loading.
Updated Nov 2025: uses `width='stretch'` instead of deprecated use_container_width.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------------------------
# Paths (update if needed)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_sets"  # change if your CSVs are elsewhere

MOVIES_CSV = DATA_DIR / "movies.csv"
RATINGS_CSV = DATA_DIR / "ratings.csv"
TAGS_CSV = DATA_DIR / "tags.csv"
LINKS_CSV = DATA_DIR / "links.csv"

# -------------------------
# TMDB config: prefer st.secrets (Streamlit Cloud), then env var
# -------------------------
def get_tmdb_key():
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("TMDB_API_KEY")

TMDB_API_KEY = get_tmdb_key()
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}"

# Fallback poster
FALLBACK_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"

# -------------------------
# Helpers: load data, features, poster fetch
# -------------------------
@st.cache_data(show_spinner=False)
def load_data():
    missing = [p for p in (MOVIES_CSV, RATINGS_CSV, TAGS_CSV, LINKS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    tags = pd.read_csv(TAGS_CSV)
    links = pd.read_csv(LINKS_CSV)
    for df in (movies, ratings, tags, links):
        if "movieId" in df.columns:
            df["movieId"] = df["movieId"].astype(int)
    if "tmdbId" in links.columns:
        links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    return movies, ratings, tags, links

@st.cache_data(show_spinner=False)
def prepare_content_features(movies_df):
    df = movies_df.copy()
    df["combined"] = df["title"].fillna("") + " " + df["genres"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()
    titles = df["title"].tolist()
    return df, cosine_sim, indices, titles

@st.cache_data(show_spinner=False)
def fetch_tmdb_poster_url(tmdb_id: int):
    if tmdb_id is None or pd.isna(tmdb_id):
        return None
    key = get_tmdb_key()
    if not key:
        return None
    url = TMDB_MOVIE_URL.format(int(tmdb_id), key)
    try:
        resp = requests.get(url, timeout=8)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    poster_path = data.get("poster_path")
    if poster_path:
        return TMDB_IMAGE_BASE + poster_path
    return None

@st.cache_data(show_spinner=False)
def recommend_indices(idx, cosine_sim, topn=10):
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1: topn + 1]
    movie_indices = [i[0] for i in sim_scores]
    return movie_indices

# -------------------------
# App UI (wrapped in try/except to avoid silent crashes)
# -------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
try:
    st.title("🎬 Content-based Movie Recommender (with Posters)")

    # TMDB info expander
    with st.expander("TMDB info (click to view)"):
        st.write("TMDB_API_KEY present?:", bool(TMDB_API_KEY))
        if TMDB_API_KEY:
            st.write("TMDB_API_KEY (first 8 chars):", TMDB_API_KEY[:8] + "...")
        st.write("If the API key is missing, set TMDB_API_KEY in your environment or Streamlit Secrets.")

    # Load data
    with st.spinner("Loading data and building features..."):
        movies, ratings, tags, links = load_data()
        movies_df, cosine_sim, indices, titles = prepare_content_features(movies)

    # Build links_map: movieId -> tmdbId (no network calls)
    links_map = {}
    if "tmdbId" in links.columns:
        tmp = links[["movieId", "tmdbId"]].dropna().drop_duplicates(subset=["movieId"])
        for _, r in tmp.iterrows():
            try:
                links_map[int(r["movieId"])] = int(r["tmdbId"])
            except Exception:
                continue

    # Search UI
    st.markdown("<h2>Movie Recommendation System</h2>", unsafe_allow_html=True)
    query = st.text_input("Search a movie (type and press Enter):", "")
    if st.button("Random"):
        import random
        query = random.choice(titles)

    selected_idx = None
    if query:
        matches_exact = movies_df[movies_df["title"].str.lower() == query.strip().lower()]
        if not matches_exact.empty:
            selected_idx = matches_exact.index[0]
        else:
            matches = movies_df[movies_df["title"].str.contains(query, case=False, na=False)]
            if not matches.empty:
                selected_idx = matches.index[0]

    if selected_idx is None:
        selected_idx = 0

    # Layout
    col_left, col_right = st.columns([1, 2])
    with col_left:
        st.subheader("Selected movie")
        selected_title = movies_df.loc[selected_idx, "title"]
        st.write(selected_title)
        movie_id = int(movies_df.loc[selected_idx, "movieId"])
        tmdb_id = links_map.get(movie_id, None)
        poster_url = None
        if tmdb_id is not None and TMDB_API_KEY:
            poster_url = fetch_tmdb_poster_url(tmdb_id)
        if poster_url:
            st.image(poster_url, width='stretch')
        else:
            if not TMDB_API_KEY:
                st.info("TMDB API key not set — posters disabled.")
            st.image(FALLBACK_POSTER, width='stretch')

    with col_right:
        st.subheader("Recommended movies")
        rec_idxs = recommend_indices(selected_idx, cosine_sim, topn=10)
        rec_cols = st.columns(5)
        for i, rec_idx in enumerate(rec_idxs):
            rec_row = movies_df.loc[rec_idx]
            rec_title = rec_row["title"]
            rec_movie_id = int(rec_row["movieId"])
            rec_tmdb = links_map.get(rec_movie_id, None)
            rec_poster = None
            if rec_tmdb is not None and TMDB_API_KEY:
                rec_poster = fetch_tmdb_poster_url(rec_tmdb)
            with rec_cols[i % 5]:
                st.caption(rec_title)
                if rec_poster:
                    st.image(rec_poster, width='stretch')
                else:
                    st.image(FALLBACK_POSTER, width=150)


except Exception as e:
    # show error in UI and prevent app from exiting silently (health check fails)
    st.error("Application error: " + str(e))
    # also log to console for Streamlit Cloud logs
    import traceback, sys
    traceback.print_exc(file=sys.stdout)

