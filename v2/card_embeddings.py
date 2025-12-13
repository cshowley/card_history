# get list of all spec_ids
from pymongo import MongoClient
import pandas as pd
import argparse
import os
from tqdm import tqdm
import math
from dotenv import load_dotenv
load_dotenv()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sentence_transformers import SentenceTransformer


# Connect using MongoDB URI
client = MongoClient(os.environ['MONGO_URI'])

# Access your database and collection
db = client["gemrate"]
collection = db["gemrate_pokemon_cards"]

pipeline = []

results = collection.aggregate(pipeline)
df = pd.DataFrame(list(results))
df = pd.json_normalize(df.to_dict('records'))
# df.to_csv('card_metadata.csv', index=None)

df = df[['CATEGORY','YEAR','SET_NAME','NAME','PARALLEL','CARD_NUMBER']]

categorical_cols = ['CATEGORY','SET_NAME','PARALLEL']

categorical_pipe = Pipeline(['ohe', OneHotEncoder])

numerical_cols = ['YEAR']
numerical_pipe = Pipeline(['scale', StandardScaler])

embed_model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    tokenizer_kwargs={"padding_side": "left"},
)

def embed_text(text):
	return embed_model.encode(text, prompt='Instruct: Given a search query, retrieve relevant passages that answer the query\nQuery:')

embedding_cols = ['NAME']
embedding_pipe = Pipeline(['embed', embed_text])

hashing_cols = ['CARD_NUMBER']

fh = FeatureHasher(n_features=2**16, input_type="string")
X = fh.transform([
    ["red"],      # sample 1 has token "red"
    ["blue"],     # sample 2 has token "blue"
    ["red"],      # sample 3 has token "red"
])
hashing_pipe = Pipeline()

preprocess = ColumnTransformer(
	[
		('categorical', categorical_pipe, categorical_cols),
		('numerical', numerical_pipe, numerical_cols),
		('embedding', embedding_pipe, embedding_cols),
		('hashing', hashing_pipe, hashing_cols),
	]
)
