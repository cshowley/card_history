"""Prediction endpoints for card price estimation."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import ModelManager, get_model_manager
from api.models import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from api.services.feature_builder import get_feature_builder

router = APIRouter(prefix="/predict", tags=["predictions"])


@router.post("/", response_model=PredictionResponse)
async def predict_price(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> PredictionResponse:
    """
    Predict the price of a Pokemon card with 95% confidence interval.

    Args:
        request: Card details including gemrate_id, grade, grading company, and source

    Returns:
        Predicted price with lower and upper bounds at 95% confidence
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Server is starting up."
        )

    try:
        # Build feature vector
        feature_builder = get_feature_builder()
        features, warnings = feature_builder.build_features(request)

        # Generate predictions
        predicted_price, lower_bound, upper_bound = model_manager.predict(features)

        return PredictionResponse(
            predicted_price=round(predicted_price, 2),
            lower_bound=round(lower_bound, 2),
            upper_bound=round(upper_bound, 2),
            confidence_level=0.95,
            warnings=warnings,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> BatchPredictionResponse:
    """
    Predict prices for multiple cards in a single request.

    Args:
        request: List of card prediction requests (max 100)

    Returns:
        List of predictions with confidence intervals
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503, detail="Models not loaded. Server is starting up."
        )

    predictions = []
    feature_builder = get_feature_builder()

    for pred_request in request.predictions:
        try:
            features, warnings = feature_builder.build_features(pred_request)
            predicted_price, lower_bound, upper_bound = model_manager.predict(features)

            predictions.append(
                PredictionResponse(
                    predicted_price=round(predicted_price, 2),
                    lower_bound=round(lower_bound, 2),
                    upper_bound=round(upper_bound, 2),
                    confidence_level=0.95,
                    warnings=warnings,
                )
            )
        except Exception as e:
            # Include error in warnings for batch, don't fail entire batch
            predictions.append(
                PredictionResponse(
                    predicted_price=0.0,
                    lower_bound=0.0,
                    upper_bound=0.0,
                    confidence_level=0.95,
                    warnings=[f"Prediction failed: {str(e)}"],
                )
            )

    return BatchPredictionResponse(
        predictions=predictions, total_processed=len(predictions)
    )
