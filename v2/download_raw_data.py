# get list of all spec_ids
from pymongo import MongoClient
import pandas as pd
import argparse
import numpy as np
import os
from tqdm import tqdm
import math
import re
from dotenv import load_dotenv
from bson import ObjectId
load_dotenv()


# Connect using MongoDB URI
client = MongoClient(os.environ['MONGO_URI'])

# Access your database and collection
db = client["gemrate"]

parser = argparse.ArgumentParser()
parser.add_argument('--include_old_data', action='store_true', default=False)
args = parser.parse_args()

collection = db["ebay_graded_items"]
output_csv_name = 'full_training_data.csv'
file_exists = os.path.isfile(output_csv_name)
if file_exists:
	os.remove(output_csv_name)

pipeline = [
    {
        "$match": {
            "gemrate_hybrid_data.specid": {"$exists": True},
            "item_data.format": "auction",
            "gemrate_hybrid_data": {"$exists": True},
            "item_data": {"$exists": True},
            "gemrate_data": {"$exists": True},
            "gemrate_data.grade": {"$exists": True},
        }
    },
    {
        "$project": {
            "gemrate_hybrid_data.specid": 1,
            "item_data.date": 1,
            "grading_company": 1,
            "gemrate_data.grade": 1,
            "item_data.price": 1,
            "item_data.number_of_bids": 1,
            "item_data.seller_name": 1,
            "item_data.best_offer_accepted": 1,
            "_id": 1 
        }
    }
]

# Save database output to dataframe
results = collection.aggregate(pipeline)
df = pd.DataFrame(list(results))
df = pd.json_normalize(df.to_dict('records'))

df = df[['_id', 'gemrate_hybrid_data.specid', 'item_data.date', "grading_company", "gemrate_data.grade", "item_data.price", "item_data.number_of_bids", "item_data.seller_name", "item_data.best_offer_accepted"]]
df.columns = ['_id','spec_id','date','grading_company', 'grade_raw', 'price', 'number_of_bids', 'seller_name', 'best_offer_accepted']

df['date'] = pd.to_datetime(df['date'])
df['price_currency'] = df['price'].str.extract(r'([^$]+)\$')[0]
df['price'] = df['price'].str.extract(r'\$([^$]*)')[0]
df['price'] = pd.to_numeric(df['price'].str.replace(',', ''), errors='coerce')

pattern = r'^(g)?(\d+(?:_\d+)?)(.*)$'

def partition_grade(grade):
    match = re.match(pattern, grade)
    if match:
        return {
            'prefix': match.group(1) or None,
            'numeric': float(match.group(2).replace('_','.')),
            'suffix': match.group(3) or None,
            'other': None
        }
    else:
        return {
            'prefix': None,
            'numeric': None,
            'suffix': None,
            'other': grade
        }

df[['grade', 'grade_suffix', 'grade_other']] = ''
for i,grade in enumerate(df.grade_raw.tolist()):
    partitioned_grades = partition_grade(grade)
    df.loc[i, 'grade'] = partitioned_grades['numeric']
    df.loc[i, 'grade_suffix'] = partitioned_grades['suffix']
    df.loc[i, 'grade_other'] = partitioned_grades['other']

if args.include_old_data:
	df['data_type'] = 'new'
	dff = pd.read_csv('../v1/full_training_data.csv')
	dff.columns = ['_id','spec_id','date','price','grade','grading_company']
	dff['data_type'] = 'old'
	df = pd.concat([df, dff], axis=0)

df.to_csv(output_csv_name, index=None)


client.close()

	# tmp = df[df['market_transaction.attributes.gradingCompany'] == 'PSA'].reset_index(drop=True)
	# tmp['market_transaction.attributes.gradeNumber'] = tmp['market_transaction.attributes.gradeNumber'].apply(lambda x: math.floor(float(x)))
	
	# pattern_price = r'\$([^$]*)'
	# df['item_data.price'] = df['item_data.price'].apply(lambda x: re.search(pattern, text))
	# pattern_currency = r'([^$]+)\$'

	# if 'item_data.item_specifics.Grade' not in df.columns:
	# 	df['item_data.item_specifics.Grade'] = np.nan
	# if 'grade' not in df.columns:
	# 	df['grade'] = np.nan
	# if 'gemrate_data.grade' not in df.columns:
	# 	df['gemrate_data.grade'] = np.nan
	
	
	# df = df[['_id', 'gemrate_data.specid', 'item_data.date', 'item_data.price', 'item_data.price_currency', 'gemrate_data.grade', 'grade_processed', 'gemrate_data.grader', 'data_type']]
	# df.columns = ['_id','spec_id','date','price','price_currency','grade_raw', 'grade_processed' 'gradingCompany', 'data_type']
	# df['price_currency'] = df.price_currency.apply(lambda x: x.strip())
	
	# # drop sales in non-US denominated currencies
	# df = df[df.price_currency == 'US']
	# # drop sales of cards with grade "auth"
	# df = df[~df.grade_processed.isna()]

	# file_exists = os.path.isfile(output_csv_name)
	# file_empty = file_exists and os.path.getsize(output_csv_name) == 0
	# df.to_csv(output_csv_name, index=None, mode='a', header=not (file_exists and not file_empty))

