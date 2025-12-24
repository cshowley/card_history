import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from inference import *
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

"""verify every entry in gemrate_pokemon_cards has an embedding
count of collection should equal count of embeddings
"""
# execute database query
client = MongoClient(os.environ['MONGO_URI'])
db = client["gemrate"]
collection = db["gemrate_pokemon_cards"]
pipeline = [
	{
		"$count": "total"
	}
]
results = collection.aggregate(pipeline)
db_count = pd.DataFrame(list(results))
db_count = pd.json_normalize(db_count.to_dict('records'))

# count number of embedding vectors
embeddings = np.load("card_vectors_768.npy").astype(np.float32)
assert int(db_count.total.iloc[0]) == embeddings.shape[0]
print('verified equal number of database entries and embeddings')


# load list of missing spec_ids
missing_spec_ids = pd.read_csv('missing.csv')
missing_spec_ids = missing_spec_ids.drop_duplicates('spec_id', keep='first')

# query database for spec_ids that were labeled missing from xgboost
pipeline = [
    {
        "$match": {
            "SPECID": {"$in": missing_spec_ids.spec_id.tolist()},
        }
    },
    {
        "$project": {
            "SPECID": 1,
            "_id": 1 
        }
    }
]


# Save database output to dataframe
results = collection.aggregate(pipeline)
df = pd.DataFrame(list(results))
df = pd.json_normalize(df.to_dict('records'))
df['SPECID'] = df['SPECID'].astype(str)
missing_spec_ids['spec_id'] = missing_spec_ids['spec_id'].astype(str)

"""
any spec ids in missing_spec_ids NOT contained in dataframe indicate:
	a) incomplete gemrate_pokemon_cards collection
	b) outdated spec_id for that pokemon card
	c) not a pokemon card and can be safely excluded
"""
# spec ids labeled missing and NOT in database
spec_ids_not_in_database = missing_spec_ids[~missing_spec_ids.spec_id.isin(df.SPECID.tolist())]

# spec ids labeled missing and IN database
spec_ids_in_database = missing_spec_ids[missing_spec_ids.spec_id.isin(df.SPECID.tolist())]


"""query embeddings table for spec_ids

any spec_ids captured in the missing_embeddings list are missing embeddings
	indicates problem with how embeddings were constructed
"""
index = CardEmbeddingIndex()
# verify that every spec_id DOES have an embedding
pipeline = [
    {
        "$project": {
            "SPECID": 1,
            "_id": 1 
        }
    }
]

# verify every spec_id in database has an embedding
results = collection.aggregate(pipeline)
all_spec_ids = pd.DataFrame(list(results))
all_spec_ids = pd.json_normalize(all_spec_ids.to_dict('records'))
# verify every spec_id in database has an embedding
missing_embeddings = []
for spec_id in tqdm(all_spec_ids.SPECID.tolist()):
    try:
        assert len(index.find_nearest(spec_id, n=3)) == 3
    except ValueError:
        missing_embeddings.append(spec_id)

if missing_embeddings:
    print(f"WARNING: {len(missing_embeddings)} spec_ids in DB are missing embeddings")
else:
    print('verified unique embedding for each database spec_id')

# look for the embeddings from the missing spec_ids list
not_in_database_with_embeddings = []
not_in_database_no_embeddings = []
for spec_id in tqdm(spec_ids_not_in_database.spec_id.tolist()):
    try:
        if len(index.find_nearest(spec_id, n=3)) == 3:
            not_in_database_with_embeddings.append(spec_id)
    except ValueError:
        not_in_database_no_embeddings.append(spec_id)

in_database_with_embeddings = []
in_database_no_embeddings = []
for spec_id in tqdm(spec_ids_in_database.spec_id.tolist()):
    try:
        if len(index.find_nearest(spec_id, n=3)) == 3:
            in_database_with_embeddings.append(spec_id)
    except ValueError:
        in_database_no_embeddings.append(spec_id)

print(f"""
ANALYSIS REPORT

Given the list of missing spec IDs
spec_ids in gemrate_pokemon_cards and have an embedding (implies error with xgboost script):	 				{len(in_database_with_embeddings)}
spec_ids  in gemrate_pokemon_cards and do not have an embedding (should be 0): 									{len(in_database_no_embeddings)}
spec_ids not in gemrate_pokemon_cards and have an embedding (should be 0): 										{len(not_in_database_with_embeddings)}
spec_ids not in gemrate_pokemon_cards and do not have an embedding (incomplete database or not pokemon cards): 	{len(not_in_database_no_embeddings)}
""")

client.close()