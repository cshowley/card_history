# get list of all spec_ids
import math
import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from tqdm import tqdm

load_dotenv()

MONGO_URL = os.getenv('MONGO_URL')
if not MONGO_URL:
    raise ValueError("MONGO_URL environment variable is not set")
client = MongoClient(MONGO_URL)

# Access your database and collection
db = client["gemrate"]
collection = db["gemrate_pokemon_cards"]

pipeline = [
	{"$project": {
		"SPECID": 1
	}},
	# {"$limit": 4}
]

results = collection.aggregate(pipeline)

df = pd.DataFrame(list(results))

df_spec_id = pd.json_normalize(df.to_dict('records'))
df_spec_id.to_csv('all_spec_ids.csv', index=None)

collection = db["alt_market_data"]
output_csv_name = 'old_market_data.csv'
file_exists = os.path.isfile(output_csv_name)
if file_exists:
	os.remove(output_csv_name)

# query in batches for each spec_id sales history
chunk_size = 8
for i in tqdm(range(0, df_spec_id.shape[0], chunk_size)):
	spec_id_chunk = df_spec_id.SPECID.iloc[i:i+chunk_size].tolist()

	pipeline = [
		{"$match": {"market_transaction.auctionType": "AUCTION"}},
		{"$match": {'market_transaction.auctionHouse': 'eBay'}},
		# {"$match": {'market_transaction.attributes.gradingCompany': 'PSA'}},
		{"$match": {"spec_id": {"$in": spec_id_chunk}}},
		{"$project": {
			"market_transaction.attributes.gradingCompany": 1,
			"market_transaction.attributes.gradeNumber": 1,
			"market_transaction.price": 1,
			"market_transaction.date": 1,
			"spec_id": 1
		}}
	]

	# Save database output to dataframe
	results = collection.aggregate(pipeline)
	df = pd.DataFrame(list(results))
	df = pd.json_normalize(df.to_dict('records'))
	if df.shape[0] == 0:
		continue


	tmp = df[df['market_transaction.attributes.gradingCompany'] == 'PSA'].reset_index(drop=True)
	tmp['market_transaction.attributes.gradeNumber'] = tmp['market_transaction.attributes.gradeNumber'].apply(lambda x: math.floor(float(x)))
	tmp['market_transaction.date'] = pd.to_datetime(tmp['market_transaction.date'])
	tmp['market_transaction.price'] = tmp['market_transaction.price'].astype(float)
	tmp.columns = ['_id','spec_id','date','price','gradeNumber','gradingCompany']
	file_exists = os.path.isfile(output_csv_name)
	file_empty = file_exists and os.path.getsize(output_csv_name) == 0
	tmp.to_csv(output_csv_name, index=None, mode='a', header=not (file_exists and not file_empty))

client.close()