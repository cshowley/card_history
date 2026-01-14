## Starting the API

```bash
# Start the server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "models_loaded": true,
  "mongodb_connected": true
}
```

### Single Prediction

**POST** `/predict/`

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "gemrate_id": "swsh4-025-normal",
    "grade": 10,
    "half_grade": false,
    "grading_company": "psa",
    "source": "ebay"
  }'
```

**Response:**

```json
{
  "predicted_price": 127.5,
  "lower_bound": 89.0,
  "upper_bound": 182.0,
  "confidence_level": 0.95,
  "warnings": []
}
```

### Batch Predictions

**POST** `/predict/batch`

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"gemrate_id": "swsh4-025-normal", "grade": 10, "half_grade": false, "grading_company": "psa", "source": "ebay"},
      {"gemrate_id": "swsh4-025-normal", "grade": 9, "half_grade": true, "grading_company": "cgc", "source": "fanatics_weekly"}
    ]
  }'
```

**Response:**

```json
{
  "predictions": [
    {
      "predicted_price": 127.5,
      "lower_bound": 89.0,
      "upper_bound": 182.0,
      "confidence_level": 0.95,
      "warnings": []
    },
    {
      "predicted_price": 85.0,
      "lower_bound": 62.0,
      "upper_bound": 115.0,
      "confidence_level": 0.95,
      "warnings": []
    }
  ],
  "total_processed": 2
}
```

## Request Parameters

| Field             | Type    | Required | Options                                                          |
| ----------------- | ------- | -------- | ---------------------------------------------------------------- |
| `gemrate_id`      | string  | Yes      | Unique card identifier from catalog                              |
| `grade`           | integer | Yes      | 1-10                                                             |
| `half_grade`      | boolean | No       | `true` for grades like 9.5, `false` otherwise (default: `false`) |
| `grading_company` | string  | Yes      | `psa`, `cgc`, `bgs`                                              |
| `source`          | string  | Yes      | `ebay`, `fanatics_weekly`, `fanatics_premier`                    |

## Interactive API Documentation

Swagger UI: `http://localhost:8000/docs`

ReDoc: `http://localhost:8000/redoc`
