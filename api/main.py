"""FastAPI application for card price prediction."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.dependencies import get_model_manager, get_neighbor_lookup
from api.models import HealthResponse
from api.routers import predict_router
from api.services.feature_builder import initialize_feature_builder
from api.services.index_cache import get_index_cache
from api.services.mongodb_client import get_mongodb_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    print("Starting Card Price Prediction API...")
    settings = get_settings()

    # Load XGBoost models
    model_manager = get_model_manager()
    model_manager.load_models()

    # Initialize feature builder with loaded feature columns
    initialize_feature_builder(model_manager.feature_cols)

    # Load neighbor lookup table
    neighbor_lookup = get_neighbor_lookup()
    neighbor_lookup.load(settings.neighbors_file)

    # Connect to MongoDB
    mongo_client = get_mongodb_client()
    try:
        mongo_client.connect()
    except Exception as e:
        print(f"Warning: MongoDB connection failed: {e}")
        print("API will run but predictions may fail without historical data.")

    # Pre-warm index cache
    index_cache = get_index_cache()
    index_cache.refresh()

    print("API startup complete!")

    yield

    # Shutdown
    print("Shutting down API...")
    mongo_client = get_mongodb_client()
    mongo_client.disconnect()
    print("API shutdown complete.")


# Create FastAPI app
settings = get_settings()
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Check API health status."""
    model_manager = get_model_manager()
    mongo_client = get_mongodb_client()

    return HealthResponse(
        status="healthy" if model_manager.is_loaded else "degraded",
        models_loaded=model_manager.is_loaded,
        mongodb_connected=mongo_client.is_connected,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
