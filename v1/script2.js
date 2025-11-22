db.alt_market_data.aggregate([
  {$match: {"market_transaction.auctionType": "AUCTION"}},
  {$match: {'market_transaction.auctionHouse': 'eBay'}},
  {$match: {"market_transaction.attributes.gradingCompany": "PSA"}},
  {$match: {"spec_id": 544028}},
//  {$limit: 1000},
  {$project: {"market_transaction.attributes.gradingCompany": 1,
              "market_transaction.attributes.gradeNumber": 1,
              "market_transaction.price": 1,
              "market_transaction.date": 1,
              "spec_id": 1}
  },
//  {$count: "total"}
])