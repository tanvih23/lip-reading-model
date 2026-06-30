import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Make from_scratch/ importable, since inference.py internally does
# `from model import LipReaderModel` and `from dummy_data import ...`
# — those only resolve if from_scratch/ itself is on the Python path.
FROM_SCRATCH_DIR = Path(__file__).resolve().parent.parent / "from_scratch"
sys.path.insert(0, str(FROM_SCRATCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference import predict_word          # Person C's function
from preprocess_upload import preprocess_upload_video  # Person A's function

app = FastAPI(title="LipSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
        temp_file.write(await video.read())

    try:
        # Person A's step: raw video -> numpy lip clip, shape (20, 96, 96)
        clip = preprocess_upload_video(str(temp_path))
        # Person C's step: numpy clip -> predicted word + confidence + all_scores
        result = predict_word(clip)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()