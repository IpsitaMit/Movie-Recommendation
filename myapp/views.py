from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Create your views here.
movies = pd.read_csv("D:\\MLarchive\\ml-move recomrnder\\movies.csv")
ratings = pd.read_csv("D:\\MLarchive\\ml-move recomrnder\\ratings.csv")
posters = pd.read_csv("D:\\MLarchive\\ml-move recomrnder\\MovieGenre.csv",encoding = "ISO-8859-1")
vectorizer = TfidfVectorizer(ngram_range = (1,2))

def cleanTitle(title):
        return re.sub("[^a-zA-Z0-9 ]","",title)

def recomender(movie_id):
        similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"]==5)]["userId"].to_numpy()
        similar_movies = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>=4)]["movieId"]
        top_similar_movies = similar_movies.value_counts() / len(similar_users) 
        top_similar_movies = top_similar_movies[top_similar_movies>0.1]
        top_allover_movies = ratings["movieId"].value_counts() / len(ratings["userId"].unique())
        top_allover_movies = top_allover_movies[top_allover_movies.index.isin(top_similar_movies.keys())]
        comparison = pd.concat([top_similar_movies,top_allover_movies], axis=1)
        comparison.columns = ["similar" , "all"]
        comparison["score"]=comparison["similar"]/comparison["all"]
        comparison=comparison.sort_values("score" ,ascending=False)
        recommended = comparison.head()
        return list(recommended.index)

def search(title,tfidf):
        title = cleanTitle(title)
        quary_vec = vectorizer.transform([title])
        similarity = cosine_similarity(quary_vec, tfidf).flatten()
        indices = np.argpartition(similarity, -3)[-3:][::-1]
        results = movies.iloc[indices]
        return results

def index(request):
    
    Movie_name = request.POST.get("Movie_name")
    if (Movie_name==None):
           Movie_name=""
    movies["clean_title"]=movies["title"].apply(cleanTitle)

    recomended_movie_names=["Type in a movie name to get recomendations!"]
    tfidf=vectorizer.fit_transform(movies["clean_title"])
    if (Movie_name!=""):
        to_match = search(Movie_name,tfidf)
        print(Movie_name)
        print("\nRecommended Movies for you :\n")
        movies_to_match=list(to_match["movieId"])
        all_recomended=[]
        for i in movies_to_match:
                all_recomended=all_recomended+recomender(i)
        all_recomended=list(set(all_recomended))
        recomended_movie_names=movies[(movies["movieId"].isin(all_recomended))]["title"].to_string(index=False).split('\n')
    

    print(recomended_movie_names)
    posters_list={}
    if recomended_movie_names[0]!="Type in a movie name to get recomendations!":
        for i in recomended_movie_names:
                final=(posters[(movies["title"]==i)]["Poster"]).to_string()
                posters_list[i]=final

        posters_list=posters_list.items()
        print(posters_list)
    context = {
          'posters_list' : posters_list,
          'movie' : recomended_movie_names
    }
    return render(request,'index.html', context)

