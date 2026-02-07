import jwt
import mlflow.pyfunc
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr , Field , validator
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime,Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from config import settings
import time
import logging


# DATABASE & ORM
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "predictions_logs"

    id = Column(Integer, primary_key=True, index=True)
    recency = Column(Float)
    frequency = Column(Integer)
    monetary = Column(Float)

    cluster_label = Column(Integer)
    cluster_probability = Column(Float)
    is_whale = Column(Boolean)
    is_noise = Column(Boolean)

    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
Base.metadata.create_all(bind=engine)


# Schema
class CustomerData(BaseModel):
    Recency: int = Field(..., gt=-1, description="Days since last purchase")
    Frequency: int = Field(..., gt=0, description="Total number of transactions")
    Monetary: float = Field(..., gt=0, description="Total money spent")

    @validator('Monetary')
    def monetary_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Monetary value must be greater than zero')
        return v

class PredictionResponse(BaseModel):
    cluster_label: int
    cluster_probability: float
    is_whale: bool
    is_noise: bool

class MultiplePredictions(BaseModel):
    predictions: List[PredictionResponse]

class ErrorResponse(BaseModel):
    detail: str


# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

try:
    model = mlflow.pyfunc.load_model(settings.MODEL_URI)
    print("✅ Model loaded successfully from Registry")
except Exception as e:
    print(f"❌ Failed to load model: {e}")


# Rate Limiting
def get_smart_identifier(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    return f"ip:{get_remote_address(request)}"


# Intialize APP
limiter = Limiter(key_func=get_smart_identifier)
app = FastAPI(title="RFM Segmentation Enterprise API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Middleware & CROS

""" 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
"""


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,  # هيقرأ القائمة اللي حددناها في الـ env
    allow_credentials=True,                 # مهم عشان الـ Tokens اللي عندك
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("RFM_system.log", encoding="utf-8"), # ضيف encoding="utf-8" هنا
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RFM_Deployment")


@app.post("/predict", response_model=MultiplePredictions)
@limiter.limit("10/minute")
async def predict_rfm(
    request: Request,
    data: List[CustomerData],
    db: Session = Depends(get_db),
):
    start_time = time.time()
    client_ip = request.client.host

    logger.info(f"📥 Incoming Request | IP: {client_ip} | Items: {len(data)}")

    try:
        input_df = pd.DataFrame([item.dict() for item in data])

        try:
            input_df['Recency'] = input_df['Recency'].astype('int64')     # تحويل لـ Long
            input_df['Frequency'] = input_df['Frequency'].astype('int64') # تحويل لـ Long
            input_df['Monetary'] = input_df['Monetary'].astype('float64') # تحويل لـ Double
        except Exception as conv_err:
            logger.error(f" Type Conversion Error: {str(conv_err)}")
            raise HTTPException(status_code=400, detail="Data type mismatch for model schema")

        logger.debug(f"Input Preview (Cast): {input_df.head(1).to_dict()}")

        preds_df = model.predict(input_df)

        log_entries = []
        for i in range(len(input_df)):
            log_entries.append(
                PredictionLog(
                    recency=float(input_df.iloc[i]['Recency']),
                    frequency=int(input_df.iloc[i]['Frequency']),
                    monetary=float(input_df.iloc[i]['Monetary']),
                    cluster_label=int(preds_df.iloc[i]['cluster_label']),
                    cluster_probability=float(preds_df.iloc[i]['cluster_probability']),
                    is_whale=bool(preds_df.iloc[i]['is_whale']),
                    is_noise=bool(preds_df.iloc[i]['is_noise']),
                    model_version="v1.0-production"
                )
            )

        try:
            db.add_all(log_entries)
            db.commit()

            process_time = round(time.time() - start_time, 4)
            logger.info(f"✅ Success | Logged: {len(log_entries)} | Latency: {process_time}s")

        except Exception as db_err:
            db.rollback()
            logger.error(f"❌ DB Failure | Error: {str(db_err)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Database insertion failed")

        return {"predictions": preds_df.to_dict(orient="records")}

    except Exception as e:
        logger.error(f"🔥 ML Model Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model Inference Error: {str(e)}")


# --- CRUD Endpoints ---
@app.get("/history") # شلنا الـ response_model عشان يبعت كل العواميد
def get_history(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).offset(skip).limit(limit).all()
    return logs


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    from sqlalchemy import func
    results = db.query(
        PredictionLog.cluster_label,
        func.count(PredictionLog.id)
    ).group_by(PredictionLog.cluster_label).all()

    return {f"Cluster {label}": count for label, count in results}

@app.delete("/history/{record_id}")
def delete_log(
        record_id: int,
        request: Request,  # ضفنا الـ request هنا عشان نجيب الـ IP
        db: Session = Depends(get_db),
):
    db_log = db.query(PredictionLog).filter(PredictionLog.id == record_id).first()
    if not db_log:
        raise HTTPException(status_code=404, detail="Record not found")

    db.delete(db_log)
    db.commit()

    client_ip = request.client.host
    logger.info(f"🗑️ Record {record_id} deleted by IP: {client_ip}")
    return {"detail": f"Record {record_id} deleted successfully"}

@app.get("/")
def health():
    return {"status": "API is running 🚀"}