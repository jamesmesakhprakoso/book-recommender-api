from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_path = "./books.csv"
try:
    data_buku_baru = pd.read_csv(data_path)
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Dataset file not found.")
except pd.errors.EmptyDataError:
    raise HTTPException(status_code=500, detail="Dataset file is empty.")

# Normalize titles for case-insensitive matching
data_buku_baru['Titles'] = data_buku_baru['Titles'].str.lower()

# Initialize FastAPI app
app = FastAPI()

# Vectorizer and Similarity Calculations
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data_buku_baru['Titles'])
similarity_matrix = cosine_similarity(tfidf_matrix)

# Similarity DataFrame
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=data_buku_baru['Titles'],
    columns=data_buku_baru['Titles']
)

# Pydantic model for API request body
class RecommendationRequest(BaseModel):
    title: str
    k: int = 5

# Function to get recommendations
def book_recommendations(title: str, k: int = 5):
    title = title.lower()  # Normalize input
    if title not in similarity_df.columns:
        raise ValueError(f"The title '{title}' does not exist in the dataset.")
    index = similarity_df.loc[:, title].to_numpy().argpartition(range(-1, -k-1, -1))
    closest_titles = similarity_df.columns[index[-(k+1):]].drop(title, errors='ignore')
    recommendations = data_buku_baru[data_buku_baru['Titles'].isin(closest_titles)].head(k)
    return recommendations.to_dict(orient='records')

# API endpoint for recommendations
@app.post("/recommend")
def recommend_books(request: RecommendationRequest):
    try:
        recommendations = book_recommendations(request.title, request.k)
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")
        return {"recommendations": recommendations}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Book Recommendation API! Visit /docs for API documentation."}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok", "total_books": len(data_buku_baru)}
