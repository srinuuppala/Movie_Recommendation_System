# streamlit_app.py
"""
Streamlit Movie Recommender with search box + heading + lazy posters.
Drop-in replacement: supports local env var TMDB_API_KEY or Streamlit Cloud st.secrets.
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
    # Streamlit Cloud: st.secrets
    try:
        if "TMDB_API_KEY" in st.secrets:
            return st.secrets["TMDB_API_KEY"]
    except Exception:
        pass
    # fallback to environment variable
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
    # check files exist
    missing = [p for p in (MOVIES_CSV, RATINGS_CSV, TAGS_CSV, LINKS_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")
    movies = pd.read_csv(MOVIES_CSV)
    ratings = pd.read_csv(RATINGS_CSV)
    tags = pd.read_csv(TAGS_CSV)
    links = pd.read_csv(LINKS_CSV)
    # normalize movieId types
    for df in (movies, ratings, tags, links):
        if "movieId" in df.columns:
            df["movieId"] = df["movieId"].astype(int)
    # normalize tmdbId to Int64 if present
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
    # also create a list of titles for quick searching
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
# App UI
# -------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
# Top heading: project name + short description
st.markdown(
    """
    <div style='display:flex; align-items:center; gap:16px'>
      <img src='https://raw.githubusercontent.com/streamlit/brand/master/logo.png' width='48' />
      <div>
        <h1 style='margin:0'>Movie Recommendation System</h1>
        <p style='margin:0; color:gray'>Content-based recommender using MovieLens + TMDB posters</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data & prepare features
with st.spinner("Loading data and building features..."):
    try:
        movies, ratings, tags, links = load_data()
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        st.stop()
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

# --- Search UI: text input + search button + optional autocomplete list ---
st.write("")  # small spacer
search_col1, search_col2, search_col3 = st.columns([4, 1, 1])
with search_col1:
    query = st.text_input("Search a movie (type a few letters and press Enter or click Search):", "")
with search_col2:
    search_btn = st.button("Search")
with search_col3:
    random_btn = st.button("Random")

# if Random clicked, pick a random movie
import random
if random_btn:
    query = random.choice(titles)

# perform search if user pressed Enter (text_input) or clicked Search
selected_idx = None
selected_title = None
if query:
    # Best-effort exact (case-insensitive) match first
    matches_exact = movies_df[movies_df["title"].str.lower() == query.strip().lower()]
    if not matches_exact.empty:
        selected_idx = matches_exact.index[0]
        selected_title = matches_exact.iloc[0]["title"]
    else:
        # contains match (first result) — behaves like "search + select first"
        matches = movies_df[movies_df["title"].str.contains(query, case=False, na=False)]
        if not matches.empty:
            selected_idx = matches.index[0]
            selected_title = matches.loc[selected_idx, "title"]

# If user clicked Search with empty query, show info
if search_btn and not query:
    st.info("Type a movie name in the search box then click Search.")

# If nothing selected yet, default to first movie in dataset
if selected_idx is None:
    selected_idx = 0
    selected_title = movies_df.loc[selected_idx, "title"]

# Layout: show selected movie on left + recommendations on right
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Searched / Selected movie")
    st.write(selected_title)
    movie_id = int(movies_df.loc[selected_idx, "movieId"])
    tmdb_id = links_map.get(movie_id, None)
    poster_url = None
    if tmdb_id is not None and TMDB_API_KEY:
        poster_url = fetch_tmdb_poster_url(tmdb_id)
    if poster_url:
        st.image(poster_url, use_container_width=True)
    else:
        if not TMDB_API_KEY:
            st.info("TMDB API key not set — posters disabled. Add TMDB_API_KEY in your environment or Streamlit Secrets.")
        st.image(FALLBACK_POSTER, use_container_width=True)

with col_right:
    st.subheader("Recommended movies")
    rec_idxs = recommend_indices(selected_idx, cosine_sim, topn=10)
    # grid of 5 columns
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
                st.image(rec_poster, use_container_width=True)
            else:
                st.image(FALLBACK_POSTER, width=150)
#st.markdown("---")
#st.write("Notes:")
#st.write("- Use the search box to find any movie and get top similar recommendations.")
#st.write("- Posters require a TMDB API key. For deployment on Streamlit Cloud, add `TMDB_API_KEY` under App → Settings → Secrets.")
#st.write("- For faster production startup, prefetch poster URLs offline and save them to a JSON that the app loads.")

#with st.expander("Debug info (click to expand)"):
 #   st.write("TMDB key present?:", bool(get_tmdb_key()))
  #  st.write("Movies shape:", movies_df.shape)
   # st.write("Ratings shape:", ratings.shape)
    #st.write("Links shape:", links.shape)
    #st.write("Selected index:", selected_idx)
    #st.write("Selected movieId:", int(movies_df.loc[selected_idx, "movieId"]))
    # show small links_map sample
    #sample_items = list(links_map.items())[:10]
    #st.write("links_map sample (movieId -> tmdbId):", sample_items)