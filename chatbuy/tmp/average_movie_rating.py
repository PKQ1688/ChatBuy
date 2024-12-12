import pandas as pd

# Load the movie data
url = 'https://phidata-public.s3.amazonaws.com/demo_data/IMDB-Movie-Data.csv'
movies_data = pd.read_csv(url)

# Calculate the average rating
average_rating = movies_data['Rating'].mean()

average_rating