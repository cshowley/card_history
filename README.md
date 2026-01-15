# Collector Crypt Data Pipeline

## Step Descriptions

- **Step 1**: Downloads sales data from eBay and PWCC via MongoDB and fetches market index data.
- **Step 2**: Generates text embeddings for catalog items using a SentenceTransformer model.
- **Step 3**: Cleans/merges sales data and generates features (previous sales, lookbacks, adjacent grades) for historical and current data.
- **Step 4**: Trains an LSTM autoencoder on normalized price histories to generate price embeddings.
- **Step 5**: Performs a nearest-neighbor search using both text and price embeddings to find similar items.
- **Step 6**: Enhances dataset with features derived from the sales history of identified nearest neighbors.
- **Step 7**: Trains multiple XGBoost models (base, lower-bound, upper-bound) in parallel using the prepared features.
- **Step 8**: Runs inference on today's data using the trained models to generate price predictions.
- **Step 9**: Sorts predictions and enforces monotonicity constraints (e.g., higher grade ≥ lower grade) within groups.
- **Step 10**: Uploads the final processed predictions to a MongoDB collection.
