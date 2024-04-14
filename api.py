from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict

class User_input(BaseModel):
    Shop : int
    BrandName : int
    ModelGroup : int
    ProductGroup : int
    day : str
    month : str
    OriginalSaleAmountInclVAT : float
    
    class Config:
        schema_extra = {
            "example": {
                "Shop": 41,
                "BrandName": -8507421977918128293,
                "ModelGroup": 3162564956579801398,
                "ProductGroup": 3162564956579801398,
                "day": "18",
                "month": "12",
                "OriginalSaleAmountInclVAT": 95.0,
            }
        }

api = FastAPI()

@api.post("/pred")
def pred(inputs: User_input):
    return predict(inputs)
 