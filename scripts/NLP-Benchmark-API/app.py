import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from hate_speech.routes.experiments import router as experiments_router

# Konfiguracja loggera root oraz Uvicorn
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

app = FastAPI(
    title="NLP Benchmark API",
    description="API do benchmarku: wykrywania mowy nienawiści z użyciem metod NLP na przetworzonym zbiorze danych.",
    version="0.1.0"
)

app.include_router(experiments_router, prefix="/experiments", tags=["experiments"])

# Globalny handler wyjątków
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception at {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error"}
    )


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"message": "Witaj w aplikacji do benchmarku NLP!"}