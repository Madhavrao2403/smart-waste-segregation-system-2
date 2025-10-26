from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from train_model import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/imageprocess")
async def process(image: UploadFile = File(...)):
    contents = await image.read()
    predicted_class , accuracy = predict_image(contents)
    accuracy = str(round(accuracy,2))
    return {"message": predicted_class+", "+accuracy+"%"}
