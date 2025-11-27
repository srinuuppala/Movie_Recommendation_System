# 🎬 Movie Recommendation System  
### Content-Based Filtering + Streamlit Web App + TMDB Posters

This project is an end-to-end **Movie Recommendation System** built using:

- **MovieLens Dataset**
- **Content-Based Filtering** (TF-IDF + Cosine Similarity)
- **Python + Scikit-Learn**
- **Interactive Streamlit App**
- **TMDB API for movie posters**
- **Lazy-loading posters for fast performance**

This is a perfect ML + NLP portfolio project that demonstrates data preprocessing, feature engineering, similarity modeling, API integration, and real-time interactive UI development.

---

## 🚀 Live Demo (Optional)
> Add your Streamlit Cloud link here after deployment:
```
🔗 https://your-app-name.streamlit.app/
```

---

## 📁 Project Structure

```
📦 Movie-Recommendation-System
 ┣ 📂 data_sets
 ┃ ┣ movies.csv
 ┃ ┣ ratings.csv
 ┃ ┣ tags.csv
 ┃ ┗ links.csv
 ┣ 📄 streamlit_app.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md  ← (this file)
 ┣ 📄 posters.json (optional if prefetching)
 ┗ 📄 .gitignore
```

---

## 🧠 Project Features

### ✔ Content-Based Recommender  
Uses **TF-IDF vectorization** on movie **title + genres** and applies **cosine similarity** to find similar movies.

### ✔ Streamlit Web Application  
Interactive web app with:
- Search box for movie titles  
- Display selected movie + poster  
- Automatically recommend top 10 similar movies  
- Beautiful UI with posters in grid layout  

### ✔ TMDB Poster Integration  
Fetches movie posters using **TMDB API** with:  
✓ Lazy loading (fast load time)  
✓ Automatic caching  
✓ Fallback poster if missing  

### ✔ Clean, optimized code  
- No heavy API calls on startup  
- Uses caching to reduce TMDB traffic  
- Production-ready for Streamlit Cloud  

---

## 📊 Example EDA Visualizations
The notebook includes:
- Ratings distribution  
- Movies per genre  
- Top-rated movies  
- Active users distribution  
- Correlation heatmap of top movies  

---

## 🛠️ Tech Stack

| Component | Technology Used |
|----------|-----------------|
| Programming | Python |
| ML Model | TF-IDF + Cosine Similarity |
| Web Framework | Streamlit |
| Data Source | MovieLens Dataset |
| Posters API | TMDB API |
| Libraries | pandas, sklearn, numpy, requests |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repo
```
git clone https://github.com/<your-username>/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 3️⃣ Add TMDB API Key

Create a `.env` file:
```
TMDB_API_KEY=your_api_key_here
```

Or set key in terminal:
```
PowerShell:
$Env:TMDB_API_KEY="your_api_key_here"
```

### 4️⃣ Run Streamlit App
```
streamlit run streamlit_app.py
```

---

## 🌐 Deploying on Streamlit Cloud

1. Upload your project to GitHub  
2. Go to: https://share.streamlit.io  
3. Create a new app  
4. Add this in **Settings → Secrets**:

```
TMDB_API_KEY = "your_api_key_here"
```

5. Deploy 🎉

---

## 🎯 Future Enhancements
- Hybrid Recommender (Content + Collaborative)
- User-based or Item-based CF
- Add movie overview descriptions from TMDB
- Add user profiles and watchlists
- Add multi-page Streamlit UI

---

## 👨‍💻 Author
**Uppala Venkata Satya Srinivas **  
- Data Science Intern  
- Aspiring Data Scientist  
- ML & NLP Enthusiast  

---

## ⭐ If you like this project  
Please ⭐ star the repository — it motivates me to create more ML projects!

