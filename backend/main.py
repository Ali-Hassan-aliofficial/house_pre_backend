from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List
import io
import traceback

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try loading the model and log any failure
try:
    model = joblib.load("house_price_model.joblib")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    traceback.print_exc()
    model = None

# Feature columns
feature_columns = [
    'Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
    'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
    'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
    'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
    'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir',
    'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
    'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu',
    'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea',
    'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold',
    'YrSold', 'SaleType', 'SaleCondition'
]

# Schema
class HouseData(BaseModel):
    Id: int
    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: int
    Street: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: float
    BsmtFinType2: str
    BsmtFinSF2: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    FirstFlrSF: float
    SecondFlrSF: float
    LowQualFinSF: float
    GrLivArea: float
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: int
    GarageFinish: str
    GarageCars: int
    GarageArea: float
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThreeSsnPorch: int
    ScreenPorch: int
    PoolArea: int
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str

@app.post("/predict")
def predict_price(data: HouseData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        input_dict = data.dict()
        print("Input data received:", input_dict)

        df = pd.DataFrame([input_dict])
        df.rename(columns={
            "FirstFlrSF": "1stFlrSF",
            "SecondFlrSF": "2ndFlrSF",
            "ThreeSsnPorch": "3SsnPorch"
        }, inplace=True)

        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_columns, fill_value=0)
        print("Processed DataFrame columns:", df.columns.tolist())

        prediction = model.predict(df)[0]
        print("Prediction:", prediction)

        return {"predicted_price": round(prediction, 2)}
    except Exception as e:
        print("Error during prediction:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        print("CSV columns before rename:", df.columns.tolist())

        df.rename(columns={
            "FirstFlrSF": "1stFlrSF",
            "SecondFlrSF": "2ndFlrSF",
            "ThreeSsnPorch": "3SsnPorch"
        }, inplace=True)

        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_columns, fill_value=0)

        predictions = model.predict(df)
        print("Predictions for CSV:", predictions)

        df["PredictedPrice"] = predictions
        return df[["PredictedPrice"]].to_dict(orient="records")
    except Exception as e:
        print("Error processing CSV:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="CSV Prediction failed")
