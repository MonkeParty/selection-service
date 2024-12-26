import numpy as np
from fastapi import FastAPI, HTTPException
from app.model import HybridRecommender
import httpx
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager
from app.config import settings
import dill as pickle
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(settings.LOG_FILENAME),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

model = None

def load_model():
    global model
    logger.info("start loading model")
    with open("app/hybrid_model.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info("model was loaded")
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Service started")
    load_model()
    yield
    logger.info("Service stopped")

app = FastAPI(lifespan=lifespan)

class UserInfo(BaseModel):
    '''
        {
            "user_id": 123,
            "avg_rating": 4.1,
            "genres_preference": {
                                    "Action": 0.8, 
                                    "Comedy": 0.6
                                },
            "tags_preference":  {
                                    "thrilling": 0.9, 
                                    "funny": 0.7
                                },
            "svd_vector": [0.62, 0.71, 0.41]
        }
    '''
    user_id : int
    avg_rating: float
    genres_preference : dict
    tags_preference : dict
    svd_vector : List[float]

@app.get("/recommend/{user_id}")
async def recommend(user_id : int):
    """
    user_id: ID пользователя
    user_ratings: Словарь {movie_id: rating} с рейтингами пользователя
    top_n: Количество фильмов для рекомендации
    """
    top_n = 10
    user_ratings = None
    recommendations = None
    logger.info(f"got request {user_id}")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.PREFERENCES_SERVICE_HOST}/preferences/{user_id}")
            response.raise_for_status()
            external_params = response.json()
    except Exception as e:
        logger.error(f"got error while try get preferences for user with id={user_id}. error: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {e}")

    try:
        user_ratings = UserInfo(**external_params).genres_preference
        recommendations = model.recommend(user_id, user_ratings, top_n=top_n)
        return {"user_id": user_id, "recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

