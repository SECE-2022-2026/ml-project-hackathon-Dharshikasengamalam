import pandas as pd
import ast
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

# Load your movie dataset
movies_df = pd.read_csv("movies.csv")

# Preprocess the genres into a list of genres
def process_genres(genre_data):
    try:
        genres = ast.literal_eval(genre_data)
        if isinstance(genres, list):
            return [str(genre) for genre in genres]
        else:
            return [str(genres)]
    except (ValueError, SyntaxError):
        return []

movies_df['genres'] = movies_df['genres'].apply(process_genres)

# One-hot encode the genres
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies_df['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Combine the genre data with other features for similarity calculation
movie_features = pd.concat([movies_df[['title']], genre_df], axis=1)

# Calculate similarity matrix
item_similarity_matrix = cosine_similarity(genre_df)

# Recommend movies based on a given title
def recommend_movies(movie_title, item_similarity_matrix):
    if movie_title not in movies_df['title'].values:
        return f"Sorry, the movie '{movie_title}' was not found in the database."

    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(item_similarity_matrix[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    similar_movies = []
    for i in range(1, 11):  # Skip the first one (the movie itself)
        movie_idx = similarity_scores[i][0]
        similar_movies.append(movies_df['title'].iloc[movie_idx])

    return similar_movies

# Get common genres, release years, and ratings
def get_common_attributes(selected_movies):
    common_genres = []
    release_years = []
    ratings = []

    for movie in selected_movies:
        movie_data = movies_df[movies_df['title'] == movie].iloc[0]
        common_genres.extend(process_genres(movie_data['genres']))
        release_years.append(str(movie_data['release_date'])[:4])
        ratings.append(movie_data.get("vote_average", 0))

    # Calculate the most common genre, year, and average rating
    common_genres = pd.Series(common_genres).mode().tolist()
    most_common_year = pd.Series(release_years).mode().iloc[0]
    avg_rating = sum(ratings) / len(ratings)

    return common_genres, most_common_year, round(avg_rating, 2)

def main():
    st.title("🎥 **Welcome to the Ultimate Movie Recommender!**")
    st.markdown("_Tell us your favorite movie, and we'll find similar ones for you!_")

    movie_title = st.text_input("Enter a movie title you like:")
    genre_filter = st.text_input("Filter by genre (optional):")
    year_filter = st.text_input("Filter by release year (optional):")
    min_rating = st.slider("Minimum rating (0-10):", 0.0, 10.0, 0.0)

    if movie_title:
        recommendations = recommend_movies(movie_title, item_similarity_matrix)

        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.write(f"Recommended movies based on '{movie_title}':")
            for movie in recommendations:
                st.write(f"- {movie}")

            selected_movie = st.selectbox("Pick a movie from the recommendations:", options=recommendations)

            if selected_movie:
                selected_movies = [movie_title, selected_movie]
                genres, year, rating = get_common_attributes(selected_movies)
                st.write("### 🎬 Common Features:")
                st.write(f"**Genres:** {', '.join(genres)}")
                st.write(f"**Year:** {year}")
                st.write(f"**Average Rating:** {rating}")

    st.markdown("_**“Movies can inspire, entertain, and connect us—enjoy your next cinematic adventure!”**_")

if __name__ == "__main__":
    main()
