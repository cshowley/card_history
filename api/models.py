"""Pydantic models for request/response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class GradingCompany(str, Enum):
    """Supported grading companies."""

    PSA = "psa"
    CGC = "cgc"
    BGS = "bgs"


class Source(str, Enum):
    """Supported sale sources."""

    EBAY = "ebay"
    FANATICS_WEEKLY = "fanatics_weekly"
    FANATICS_PREMIER = "fanatics_premier"


class PredictionRequest(BaseModel):
    """Request schema for price prediction."""

    gemrate_id: str = Field(..., description="Unique card identifier")
    grade: int = Field(..., ge=1, le=10, description="Card grade (1-10)")
    half_grade: bool = Field(
        False, description="Whether grade includes 0.5 (e.g., 9.5)"
    )
    grading_company: GradingCompany = Field(..., description="Grading company")
    source: Source = Field(..., description="Sale source/marketplace")

    class Config:
        json_schema_extra = {
            "example": {
                "gemrate_id": "abc123",
                "grade": 10,
                "half_grade": False,
                "grading_company": "psa",
                "source": "ebay",
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for price prediction."""

    predicted_price: float = Field(..., description="Point prediction (USD)")
    lower_bound: float = Field(..., description="Lower bound of 95% interval (USD)")
    upper_bound: float = Field(..., description="Upper bound of 95% interval (USD)")
    confidence_level: float = Field(
        0.95, description="Confidence level for the interval (always 95%)"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Any warnings about data quality"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_price": 127.50,
                "lower_bound": 89.00,
                "upper_bound": 182.00,
                "confidence_level": 0.95,
                "warnings": [],
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    predictions: list[PredictionRequest] = Field(
        ..., max_length=100, description="List of prediction requests (max 100)"
    )


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: list[PredictionResponse]
    total_processed: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether all models are loaded")
    mongodb_connected: bool = Field(..., description="MongoDB connection status")
