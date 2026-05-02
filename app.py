from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import json
from transformers import AutoTokenizer, CamembertForSequenceClassification

# ============================================
# CONFIGURATION
# ============================================

MODEL_HF_REPO   = "greykingreys/medconnect-camembert"
DEPLOYMENT_PATH = "/app"
MAX_LENGTH      = 128
SEUIL_CONFIANCE = 0.3

# ============================================
# CHARGEMENT DU MODELE
# ============================================

print("Chargement du modele...")

tokenizer     = AutoTokenizer.from_pretrained(MODEL_HF_REPO)
model         = CamembertForSequenceClassification.from_pretrained(MODEL_HF_REPO)
label_encoder = joblib.load(f"{DEPLOYMENT_PATH}/label_encoder.joblib")

with open(f"{DEPLOYMENT_PATH}/reponses_par_maladie.json", "r", encoding="utf-8") as f:
    reponses_par_maladie = json.load(f)

with open(f"{DEPLOYMENT_PATH}/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)
model.eval()

print(f"Modele charge sur : {DEVICE}")

# ============================================
# SCHEMAS
# ============================================

class RequeteSymptomes(BaseModel):
    symptomes: str

class ReponseModele(BaseModel):
    maladie   : str
    reponse   : str
    confiance : float
    certain   : bool

# ============================================
# APPLICATION
# ============================================

app = FastAPI(
    title       = "MedConnect AI API",
    description = "API de prediction de maladies basee sur les symptomes",
    version     = "1.0.0"
)

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def accueil():
    return {
        "status"   : "ok",
        "modele"   : config["model_name"],
        "maladies" : config["num_labels"],
        "accuracy" : config["best_accuracy"]
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predire", response_model=ReponseModele)
def predire(requete: RequeteSymptomes):
    if not requete.symptomes.strip():
        raise HTTPException(status_code=400, detail="Les symptomes ne peuvent pas etre vides")

    encoding = tokenizer(
        requete.symptomes,
        truncation     = True,
        padding        = True,
        max_length     = MAX_LENGTH,
        return_tensors = "pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs    = model(**encoding)
        probs      = torch.softmax(outputs.logits, dim=1)
        confiance  = probs.max().item()
        pred_label = probs.argmax().item()

    maladie = label_encoder.inverse_transform([pred_label])[0]

    if confiance < SEUIL_CONFIANCE:
        return ReponseModele(
            maladie   = "Inconnu",
            reponse   = "Je ne suis pas certain. Veuillez consulter un professionnel de sante.",
            confiance = round(confiance, 2),
            certain   = False
        )

    reponse = reponses_par_maladie.get(
        maladie,
        "Consultez un professionnel de sante."
    )

    return ReponseModele(
        maladie   = maladie,
        reponse   = reponse,
        confiance = round(confiance, 2),
        certain   = True
    )