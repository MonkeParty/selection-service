import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import dill as pickle

class HybridRecommender:
    def __init__(self, svd_model, movie_genres, unique_genres):
        """
        Инициализация модели.
        :param svd_model: Обученная модель SVD.
        :param movie_genres: DataFrame с жанрами фильмов.
        :param unique_genres: Список уникальных жанров.
        """
        self.svd_model = svd_model
        self.movie_genres = movie_genres
        self.unique_genres = unique_genres

    def get_user_content_preferences(self, user_ratings):
        import numpy as np
        """
        Рассчитывает предпочтения пользователя на основе жанров.
        :param user_ratings: Словарь {movie_id: rating}.
        :return: Вектор предпочтений пользователя.
        """
        user_vector = np.zeros(len(self.unique_genres))
        for movie_id, rating in user_ratings.items():
            if movie_id in self.movie_genres['movieId'].values:
                genre_vector = self.movie_genres[self.movie_genres['movieId'] == movie_id].iloc[:, 1:].values.flatten()
                user_vector += rating * genre_vector
        return user_vector / np.sum(user_vector) if np.sum(user_vector) > 0 else user_vector

    def predict(self, user_id, user_ratings, movie_id, weight_collab=0.7, weight_content=0.3):
        import numpy as np
        """
        Предсказывает рейтинг для фильма.
        :param user_id: ID пользователя.
        :param user_ratings: Словарь {movie_id: rating}.
        :param movie_id: ID фильма.
        :param weight_collab: Вес коллаборативной фильтрации.
        :param weight_content: Вес контентной фильтрации.
        :return: Прогнозируемый рейтинг.
        """
        # Предсказание коллаборативной фильтрации
        collab_score = self.svd_model.predict(user_id, movie_id).est

        # Предсказание на основе контентной модели
        user_content_preferences = self.get_user_content_preferences(user_ratings)
        if movie_id in self.movie_genres['movieId'].values:
            genre_vector = self.movie_genres[self.movie_genres['movieId'] == movie_id].iloc[:, 1:].values.flatten()
            content_score = np.dot(user_content_preferences, genre_vector)
        else:
            content_score = 0  # Если жанров нет, контентный вклад отсутствует

        # Итоговое предсказание
        return weight_collab * collab_score + weight_content * content_score

    def recommend(self, user_id, user_ratings, top_n=10, weight_collab=0.7, weight_content=0.3):
        """
        Рекомендует топ-N фильмов для пользователя.
        :param user_id: ID пользователя.
        :param user_ratings: Словарь {movie_id: rating}.
        :param top_n: Количество рекомендаций.
        :param weight_collab: Вес коллаборативной фильтрации.
        :param weight_content: Вес контентной фильтрации.
        :return: Список рекомендаций.
        """
        # Собираем список всех фильмов
        all_movies = set(self.movie_genres['movieId'])

        # Исключаем фильмы, которые пользователь уже оценил
        watched_movies = set(user_ratings.keys())
        candidate_movies = all_movies - watched_movies

        # Рассчитываем прогнозируемые рейтинги для каждого фильма
        recommendations = []
        for movie_id in candidate_movies:
            predicted_score = self.predict(user_id, user_ratings, movie_id, weight_collab, weight_content)
            recommendations.append((movie_id, predicted_score))

        # Сортируем по прогнозируемому рейтингу в порядке убывания
        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        # Возвращаем топ-N фильмов
        return recommendations[:top_n]

if __name__ == "__main__":
    from surprise import SVD, Dataset, Reader

    # Загружаем данные
    ratings = pd.read_csv("app/source/rating.csv")
    movies = pd.read_csv("app/source/movie.csv")

    # Обрабатываем жанры
    movies['genres'] = movies['genres'].str.split('|')
    unique_genres = sorted(set(genre for sublist in movies['genres'] for genre in sublist))
    genre_matrix = pd.DataFrame([
        [1 if genre in genres else 0 for genre in unique_genres] for genres in movies['genres']
    ], columns=unique_genres)

    # Объединяем фильмы и жанры
    movie_genres = pd.concat([movies[['movieId']], genre_matrix], axis=1)

    # Подготовка данных для Surprise
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Обучение модели SVD
    svd_model = SVD()
    svd_model.fit(data.build_full_trainset())

    # Создаем и сохраняем гибридную модель
    hybrid_model = HybridRecommender(svd_model, movie_genres, unique_genres)

    with open("hybrid_model.pkl", "wb") as f:
        pickle.dump(hybrid_model, f)