db.getCollection("gemrate_pokemon_cards").find({})

db.getCollection("gemrate_pokemon_cards").aggregate([{"$match": {"SPECID": 3700060}}])