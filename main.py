from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

app = FastAPI()

MODEL_PATH = "model/model_covid.pth"
CLASSES = ['Covid', 'Normal', 'Viral Pneumonia']

# Transformação igual à do treino
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# Carrega o modelo
def carregar_modelo():
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, len(CLASSES))
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model = carregar_modelo()


# ─────────────────────────────────────────────
# Funcionalidade 02, 03 e 04 — Previsão completa
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas imagens são permitidas.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao processar a imagem.")

    img_tensor = transform_val(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(img_tensor)
        probas = torch.softmax(logits, dim=1)
        top_idx = probas.argmax().item()
        confidence = probas[0, top_idx].item()

    pred_class = CLASSES[top_idx]
    tem_pneumonia = pred_class != "Normal"

    if pred_class == "Normal":
        tipo = "Sem pneumonia (pulmão saudável)"
    elif pred_class == "Viral Pneumonia":
        tipo = "Pneumonia viral"
    else:
        tipo = "Covid-19 (possivelmente viral)"

    return JSONResponse({
        "diagnostico": tipo,
        "tem_pneumonia": tem_pneumonia,
        "classe_predita": pred_class,
        "confianca": round(confidence, 4)
    })


# ─────────────────────────────────────────────
# Funcionalidade 05 — Health check
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok", "message": "API online"})
