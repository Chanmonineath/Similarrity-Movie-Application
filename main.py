import streamlit as st
from pyspark import SparkConf, SparkContext
import pandas as pd
import math
from streamlit_option_menu import option_menu

# Initialize Spark Context
def init_spark():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
    sc = SparkContext.getOrCreate(conf=conf)
    return sc

# Load Movie Names
def load_movie_names():
    movie_names = {}
    with open("ml-100k/u.item", encoding='ISO-8859-1') as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

# Load Ratings Data
def load_ratings_data(sc):
    lines = sc.textFile("ml-100k/u.data")
    ratings = lines.map(lambda x: x.split()).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))))
    return ratings

# Make Pairs
def make_pairs(user_ratings):
    (user, ratings) = user_ratings
    ratings = list(ratings)
    pairs = []
    for i in range(len(ratings)):
        for j in range(i + 1, len(ratings)):
            pairs.append(((ratings[i][0], ratings[j][0]), (ratings[i][1], ratings[j][1])))
    return pairs

# Filter Duplicates
def filter_duplicates(movie_pair):
    movie1, movie2 = movie_pair[0]
    return movie1 < movie2

# Compute Cosine Similarity
def compute_cosine_similarity(rating_pairs):
    num_pairs = 0
    sum_xx = sum_yy = sum_xy = 0
    for ratingX, ratingY in rating_pairs:
        sum_xx += ratingX * ratingX
        sum_yy += ratingY * ratingY
        sum_xy += ratingX * ratingY
        num_pairs += 1

    denominator = math.sqrt(sum_xx) * math.sqrt(sum_yy)
    if denominator == 0:
        score = 0
    else:
        score = sum_xy / float(denominator)
    
    return (score, num_pairs)

# Find Similar Movies
def find_similar_movies(sc, score_threshold=0.97, co_occurrence_threshold=50):
    ratings = load_ratings_data(sc)
    ratings_by_user = ratings.groupByKey()
    movie_pairs = ratings_by_user.flatMap(make_pairs)
    filtered_movie_pairs = movie_pairs.filter(filter_duplicates)
    movie_pair_ratings = filtered_movie_pairs.groupByKey()
    movie_pair_similarities = movie_pair_ratings.mapValues(compute_cosine_similarity).cache()
    
    filtered_results = movie_pair_similarities.filter(
        lambda pairSim: pairSim[1][0] > score_threshold and pairSim[1][1] > co_occurrence_threshold
    )
    
    results = filtered_results.map(lambda pairSim: (pairSim[1], pairSim[0])).sortByKey(False)
    return results

# Get Similar Movies
def get_similar_movies(movie_id, results, movie_names, top_n=10):
    top_similar_movies = []
    for result in results.take(top_n):
        (sim, pair) = result
        similar_movie_id = pair[1] if pair[0] == movie_id else pair[0]
        top_similar_movies.append({
            'Movie ID': similar_movie_id,
            'Movie Name': movie_names[similar_movie_id],
            'Similarity Score': sim[0],
            'Co-occurrence': sim[1]
        })
    return top_similar_movies

# Ensure the SparkContext is stopped when Streamlit exits
def cleanup():
    try:
        sc = SparkContext.getOrCreate()
        if sc is not None:
            sc.stop()
    except Exception as e:
        pass

import atexit
atexit.register(cleanup)

# User Login
def login():
    st.sidebar.title("Login")
    st.sidebar.markdown("<style>.sidebar .sidebar-content { padding: 1rem; }</style>", unsafe_allow_html=True)
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if email and password:  # Add real authentication here
            st.session_state["email"] = email
            st.sidebar.success(f"Logged in as {email}")
        else:
            st.sidebar.error("Please enter a valid email and password")

# Apply custom CSS
def apply_css():
    st.markdown("""
        <style>
        body {
            color: black;
        }
        .sidebar .sidebar-content {
            background-color: #ffcccb;
            padding: 1rem;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: #ff69b4;
            color: black;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #ff1493;
        }
        .stTextInput>div>div>input {
            border-radius: 5px;
            border: 1px solid #ff69b4;
            padding: 0.5rem;
        }
        .stTextInput>div>div>input:focus {
            border-color: #ff1493;
            outline: none;
            box-shadow: 0 0 5px rgba(255, 20, 147, 0.5);
        }
        .stSlider>div>div>div>div {
            background: #ff69b4;
        }
        .st-d4, .st-d6, .st-d3, .st-d5 {
            background-color: #ffb6c1;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Main UI
def main():
    st.set_page_config(page_title="Movie Similarity AI Application", layout="wide")
    st.markdown("""
        <style>
        .main {
            background-color: #79305a;
        }
        </style>
        """, unsafe_allow_html=True)

    apply_css()

    if "email" not in st.session_state:
        login()
    else:
        with st.sidebar:
            selected = option_menu(
                menu_title="Setting",
                options=["Home Page", "Find Similar Movies", "Instruction"],
                icons=["house", "search", "info-circle"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "Home Page":
            st.title("🎬 Movie Similarity AI Application")
            
            st.markdown("""
            ## 🥰Welcome to the Movie Similarity Finder AI Application!🥰
            
            """)

            st.markdown("""
            ### ❤️Discover New Movies❤️
            Immerse yourself in a wide selection of films from different genres and eras. Whether you're a dedicated movie buff or a casual watcher, our aim is to enrich your viewing experience with customized suggestions that align with your personal taste.
            """)

            

            st.markdown("""
            ### 🙌How It Works
            Our system analyzes user ratings to find movies that are similar to the ones you love. By leveraging the MovieLens dataset, we ensure that the recommendations are based on real user preferences, making them highly accurate and tailored to your taste.
            """)

            st.markdown("""
            ### 🤔Why Use Our App?
            - **Personalized Recommendations:** Get movie suggestions based on your favorite films.
            - **Wide Selection:** Discover movies from various genres and eras.
            - **Easy to Use:** Simply select a movie and find similar recommendations in seconds.
            """)

            st.markdown("""
            ### 😁Get Started
            Head over to the "Find Similar Movies" section, select a movie, adjust the thresholds if needed, and click on "Find Similar Movies" to get started with your personalized movie recommendations.
            """)

        if selected == "Find Similar Movies":
            st.title("🔍 Find Similar Movies")
            st.markdown("### Select a movie and find its most similar movies based on user ratings.")
            sc = init_spark()
            if sc is not None:
                movie_names = load_movie_names()

                selected_movie_name = st.selectbox("Select a Movie", options=list(movie_names.values()))
                movie_id = next(key for key, value in movie_names.items() if value == selected_movie_name)
                score_threshold = st.slider("Score Threshold", 0.0, 1.0, 0.97)
                co_occurrence_threshold = st.slider("Co-occurrence Threshold", 1, 100, 50)

                if st.button("Find Similar Movies"):
                    with st.spinner('Finding similar movies...'):
                        results = find_similar_movies(sc, score_threshold, co_occurrence_threshold)
                        similar_movies = get_similar_movies(movie_id, results, movie_names)
                    
                    st.success(f"Top 10 similar movies for {movie_names[movie_id]}:")
                    st.write("Here are the top similar movies based on the user ratings:")

                    if similar_movies:
                        df = pd.DataFrame(similar_movies)
                        st.dataframe(df)
                    else:
                        st.write("No similar movies found. Try adjusting the thresholds.")
            else:
                st.error("Failed to initialize SparkContext. Please check your Spark installation.")

        if selected == "Instruction":
            st.title("Instructions for Using the Movie Similarity Finder Application")
            st.header("Welcome to the Movie Similarity Finder Webapp")
            st.markdown("""
                This application allows you to find movies similar to your favorite ones based on user ratings.

                ### How to Use:

                **Step 1:** Go to the "Find Similar Movies" section.

                **Step 2:** Select a movie from the dropdown menu. The list contains all available movies in the dataset.

                **Step 3:** Adjust the **Score Threshold** using the slider. This determines the minimum similarity score for the recommendations (default is 0.97).

                **Step 4:** Adjust the **Co-occurrence Threshold** using the slider. This sets the minimum number of co-ratings required for the recommendations (default is 50).

                **Step 5:** Click the "Find Similar Movies" button to start the search.

                The app will display the top 10 movies similar to the selected movie, along with their similarity scores and co-occurrence counts.
            """)

           
if __name__ == "__main__":
    main()
