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
collection = db["gemrate_pokemon_cards"]

parser = argparse.ArgumentParser()
parser.add_argument('--include_old_data', action='store_true', default=False)
args = parser.parse_args()

print(args.include_old_data)

pipeline = [
	{"$project": {
		"SPECID": 1
	}},
	# {"$limit": 4}
]

spec_id_lookup = 'all_spec_ids.csv'
if not os.path.exists(spec_id_lookup):
	results = collection.aggregate(pipeline)

	df = pd.DataFrame(list(results))

	df_spec_id = pd.json_normalize(df.to_dict('records'))
	df_spec_id.to_csv('all_spec_ids.csv', index=None)
else:
	df_spec_id = pd.read_csv(spec_id_lookup)

df_spec_id.SPECID = df_spec_id.SPECID.astype(str)

collection = db["ebay_graded_items"]
output_csv_name = 'full_training_data.csv'
file_exists = os.path.isfile(output_csv_name)
if file_exists:
	os.remove(output_csv_name)

# query in batches for each spec_id sales history
chunk_size = 8
# id_list = ['68ebd5af0047dc1fd31aa09d', '68ebd49e0047dc1fd314c6d4', '68ebd6af0047dc1fd3205dc4', '68ebd49f0047dc1fd314d206', '68ebd5710047dc1fd3193d1f', '68ebd5f10047dc1fd31baf93', '68ebd5f10047dc1fd31bb717', '68ebd5f10047dc1fd31bb945', '68ebd5fc0047dc1fd31c5f4f', '68ebd6de0047dc1fd321b5fb', '68ee964080837fcf1f1b1bb4', '68efd89b5f0ab4c338bd6f0b', '68ebd5ef0047dc1fd31b9ec8', '68f548acb06e15f3bb55d0c9', '68ebd65d0047dc1fd31e0f23', '68ffabbdb010a124c41d6ea6', '68ebd72e0047dc1fd322f7e3', '68ebd6a60047dc1fd31fc785', '6914bbb24cb7052a1f330605', '68ebd4a90047dc1fd314f9d6', '68ebd75d0047dc1fd323b1bc', '690e173d3636617d184ed541', '68edd0f57eca2d16a583c6b7', '68ebd5d50047dc1fd31b600e', '68ebd5d50047dc1fd31b600d', '690b7c39020a3fb8b5a42bb6', '68ffabc0b010a124c41d8846', '68ebd4a90047dc1fd314f301', '68ebd4c80047dc1fd3157881', '690b7c39020a3fb8b5a42cf3', '68ebd6d40047dc1fd3211356', '68ebd4c80047dc1fd31576dd', '68ebd4c80047dc1fd31576b2', '68f6511472ffb082a3b29105', '690a3b6c095c18e50faff149', '68ebd4c80047dc1fd31576a7', '68ebd49f0047dc1fd314d5bd', '68ebd5fb0047dc1fd31c45d3', '68ebd6630047dc1fd31e7e7c', '68ebd6ad0047dc1fd3203fc0', '68f548b2b06e15f3bb55f63f', '68ebd5700047dc1fd3193660', '68ebd5c90047dc1fd31b524a', '68ebd5d70047dc1fd31b7f46', '68ebd4ce0047dc1fd3158663', '68ebd4c80047dc1fd3157646', '690cbf1176e44d7ee615a285', '6915f1b46a0ba9a2b33e5b18', '68ebd5d60047dc1fd31b6dbb', '68ebd4c80047dc1fd31576e3', '6908d2a92191c1396a2d0104', '690e173d3636617d184ed545', '68ebd5f10047dc1fd31bb006', '68ebd4c70047dc1fd3156e73', '6905006ca2243b99c3e51174', '68ebd6270047dc1fd31d4aea', '68ebd7360047dc1fd32302ac', '68ebd72d0047dc1fd322ea8b', '68ebd5fc0047dc1fd31c609c', '68ebd70c0047dc1fd3226e35', '690a3b6b095c18e50fafe48e', '6903b2488aa0cdced74d782b', '68ffabc6b010a124c41da1b7', '68ebd66e0047dc1fd31f1f09', '68ebd58a0047dc1fd319ac47', '68ebd6ec0047dc1fd321bf20', '68ebd72d0047dc1fd322e126', '690a3b6c095c18e50faff1ae', '68ebd58b0047dc1fd319bddc', '68edd0f57eca2d16a583c367', '68ebd5d50047dc1fd31b5fca', '68ebd5090047dc1fd317380b', '68ebd58b0047dc1fd319c72e', '68ebd6270047dc1fd31d5512', '68f6511472ffb082a3b29513', '68ffabbfb010a124c41d7e4b', '68f6511472ffb082a3b29519', '68ebd49f0047dc1fd314d5f5', '68ffabbfb010a124c41d7e3e', '68edd0f47eca2d16a583c0b9', '6908d2aa2191c1396a2d080d', '68ffabbfb010a124c41d81b8', '68ebd5fd0047dc1fd31c6762', '6908d2aa2191c1396a2d0801', '68ebd5340047dc1fd3185695', '68ebd4ed0047dc1fd315f4f4', '68ffabbfb010a124c41d7cc4', '68ebd58f0047dc1fd319fb21', '68ebd5410047dc1fd31890bf', '68ebd5d70047dc1fd31b7cfc', '68fbb12ce87cd8adc3757030', '68ebd5340047dc1fd31855ec', '68ebd5c70047dc1fd31b3221', '68efd89c5f0ab4c338bd805a', '68ebd6690047dc1fd31ed530', '690e173a3636617d184ec414', '68ebd7180047dc1fd3229e13', '68ebd5c80047dc1fd31b3f84', '68ebd5c80047dc1fd31b3f98', '68ebd56f0047dc1fd31925c8', '68ee964680837fcf1f1b3f56', '68ebd5410047dc1fd31890cd', '68f6511372ffb082a3b28326', '68ebd5a70047dc1fd31a2a1f', '68ebd5c70047dc1fd31b3ef3', '68f6511472ffb082a3b29666', '68ffabbfb010a124c41d81f3', '68ebd72e0047dc1fd322f013', '68ebd5c80047dc1fd31b4971', '68f6511472ffb082a3b29515', '68ebd6600047dc1fd31e4956', '690a3b6c095c18e50faff250', '68f6511b72ffb082a3b2a32e', '68ebd65d0047dc1fd31e1342', '68ebd72e0047dc1fd322f281', '68ebd5560047dc1fd318df45', '68ebd5560047dc1fd318ead6', '68ebd6af0047dc1fd3205c25', '68ebd66f0047dc1fd31f2e57', '68ebd6b00047dc1fd3206648', '68ebd5860047dc1fd3197a84', '68ebd6640047dc1fd31e860b', '68ebd5d70047dc1fd31b75d9', '68ebd6f10047dc1fd321fd94', '68ebd6f10047dc1fd32202f3', '68ebd7420047dc1fd323352d', '68ebd5c90047dc1fd31b5281', '68ffabbfb010a124c41d7e69', '68ebd56f0047dc1fd31926c7', '68ffabc0b010a124c41d8a47', '68ebd7020047dc1fd3223ac2', '68ebd6ac0047dc1fd3202605', '68ebd5710047dc1fd3193d2b', '68ebd6f10047dc1fd32202da', '68ebd72e0047dc1fd322f728', '690798b8569e201aa15d7bc9', '691c36e5217daeb3cfefaa8a', '68ebd65d0047dc1fd31e1465', '68ebd58b0047dc1fd319ba16', '68ebd5c60047dc1fd31b2cfc', '68ebd72e0047dc1fd322f211', '68ebd56f0047dc1fd319266d', '68ebd5880047dc1fd3198f3d', '68ebd58c0047dc1fd319d3fc', '68efd89c5f0ab4c338bd8082', '68ebd6d30047dc1fd3210cfe', '68ebd5880047dc1fd31999d8', '68efd89a5f0ab4c338bd69a8', '68ebd6b30047dc1fd32092b7', '690a3b6c095c18e50faff13a', '68ebd5170047dc1fd3178aaf', '68fcf35a214c351c739b2b39', '68fcf35a214c351c739b27e1', '68fe30e456abd4f8cb912f5c', '68ebd67f0047dc1fd31f9975', '68f6511272ffb082a3b2715d', '6908d2aa2191c1396a2d105d', '68ebd5d70047dc1fd31b83e5', '68fbb130e87cd8adc375a0ac', '68ebd5090047dc1fd3173388', '6914bbab4cb7052a1f32facf', '68ebd4a90047dc1fd314ecd9', '68f7be2fc81e8b10c1f62d7f', '68ebd51a0047dc1fd317d53b', '68ebd6a60047dc1fd31fd2c2', '68ebd6d40047dc1fd3211296', '68f3bad3ab3484bca0d5e048', '68ebd6a60047dc1fd31fcb5f', '68ffabbfb010a124c41d7fa8', '68ebd6d20047dc1fd320f652', '6914c196c869f2bce5737a68', '68f548afb06e15f3bb55ebe0', '6908d2ab2191c1396a2d14cf', '6915f1a16a0ba9a2b33e15dd', '68ebd4a00047dc1fd314e3fd', '6915f1b46a0ba9a2b33e536c', '68ebd6d90047dc1fd32172dd', '68ebd6ab0047dc1fd320187a', '68ebd6ab0047dc1fd3201878', '68ebd7370047dc1fd32317b4', '68ebd6590047dc1fd31dde71', '6914c197c869f2bce5737c88', '68fbb130e87cd8adc375a4d7', '690798b8569e201aa15d7bf2', '68ebd5410047dc1fd3188e6a', '68f6511572ffb082a3b296af', '68ebd5f10047dc1fd31bb890', '68f6511472ffb082a3b2951a', '68fcf353214c351c739ad5d6']

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

