from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from predict import predict_image

app = FastAPI()

# 允許跨網域存取（給前端用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_image(contents)
    return {"result": result}