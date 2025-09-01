from pymongo import MongoClient
import pandas as pd
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument('spec_id')
args = parser.parse_args()

print(args)
# Connect using MongoDB URI
client = MongoClient(os.environ['MONGO_URI'])

# Access your database and collection
db = client["gemrate"]
collection = db["alt_market_data"]
spec_id = int(args.spec_id)

pipeline = [
    {"$match": {"market_transaction.auctionType": "AUCTION"}},
    {"$match": {'market_transaction.auctionHouse': 'eBay'}},
    {"$match": {"spec_id": spec_id}},
    {"$project": {
        "market_transaction.attributes.gradingCompany": 1,
        "market_transaction.attributes.gradeNumber": 1,
        "market_transaction.price": 1,
        "market_transaction.date": 1,
        "spec_id": 1
    }}
]

results = collection.aggregate(pipeline)

# THE MAGIC HAPPENS HERE:
df = pd.DataFrame(list(results))

# Handle nested structures automatically
df = pd.json_normalize(df.to_dict('records'))

# Close connection when done (critical for scripts)
client.close()

### Plot results
tmp = df[df['market_transaction.attributes.gradingCompany'] == 'PSA'].reset_index(drop=True)
import matplotlib.pyplot as plt
plt.figure()
tmp['market_transaction.attributes.gradeNumber'] = tmp['market_transaction.attributes.gradeNumber'].astype(float)
for grade in sorted(tmp['market_transaction.attributes.gradeNumber'].unique()):
    tmp_ = tmp[tmp['market_transaction.attributes.gradeNumber'] == grade]
    tmp_ = tmp_.sort_values(by='market_transaction.date')
    tmp_['market_transaction.date'] = pd.to_datetime(tmp_['market_transaction.date'])
    tmp_['market_transaction.price'] = tmp_['market_transaction.price'].astype(float)
    plt.plot(tmp_['market_transaction.date'], tmp_['market_transaction.price'], label=grade)
    plt.legend(loc='best')
    plt.xlabel('Sales date')
    plt.ylabel('Sales price')
    plt.title(f'spec_id {spec_id}')
    plt.xticks(rotation=45)
plt.savefig(f'{spec_id}.png')
plt.close('all')
