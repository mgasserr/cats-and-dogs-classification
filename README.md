# Cat vs Dog Classifier — Website

A simple full-stack app: upload a photo in the browser and a FastAPI backend
runs it through the VGG16 transfer-learning model (same architecture as your
notebook) to say cat or dog, with a confidence score.

```
cat-dog-classifier/
├── backend/
│   ├── main.py            # FastAPI server, POST /predict
│   ├── train.py           # Trains the model from the notebook, saves weights
│   └── requirements.txt
├── frontend/
│   └── index.html         # React app (no build step needed)
└── README.md
```

## 1. Install backend dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a trained model

The backend expects a trained model file at `backend/model/cat_dog_vgg16.h5`.
You have two options:

**Option A — train it yourself** (reproduces the notebook exactly):

```bash
python train.py
```

This downloads the `tongpython/cat-and-dog` dataset via `kagglehub` (you'll
need a Kaggle account the first time — kagglehub will prompt you, or you can
place a `kaggle.json` API token in `~/.kaggle/`), trains for 5 epochs, and
saves the model to `backend/model/cat_dog_vgg16.h5`.

**Option B — already have a trained `.h5` file** (e.g. saved from Colab)?
Copy it to `backend/model/cat_dog_vgg16.h5`, or point the server at it with
an environment variable:

```bash
export MODEL_PATH=/path/to/your/model.h5
```

## 3. Run the backend

```bash
uvicorn main:app --reload --port 8000
```

Check it's alive: open `http://localhost:8000/health` — it should say
`"model_ready": true` once the model file is in place.

## 4. Open the frontend

Just open `frontend/index.html` directly in your browser (double-click it,
or `open frontend/index.html` / drag it into a browser tab). No build step,
no Node install — it's a single file using React from a CDN.

Upload a photo of a cat or dog and it will call the backend at
`http://localhost:8000/predict` and show the result.

> If your backend runs somewhere other than `localhost:8000`, edit the
> `API_URL` constant near the top of the `<script type="text/babel">` block
> in `index.html`.

## Notes

- The model architecture and preprocessing in `train.py` and `main.py`
  exactly match your notebook: VGG16 (ImageNet weights, frozen, no top),
  `vgg16.preprocess_input` applied inside the model via a `Lambda` layer,
  `Flatten → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)`, with
  sigmoid output `> 0.5` meaning dog.
- This is a demo-grade setup (CORS wide open, single worker) — fine for local
  use, but harden it (proper CORS origins, auth, a production ASGI setup)
  before deploying publicly.
