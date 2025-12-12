# get list of all spec_ids
from pymongo import MongoClient
import pandas as pd
import argparse
import os
from tqdm import tqdm
import math
from dotenv import load_dotenv
load_dotenv()


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
numerical_cols = ['YEAR']
embedding_cols = ['NAME','CARD_NUMBER']