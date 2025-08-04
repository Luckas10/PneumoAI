from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import io
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

app = FastAPI()

MODEL_PNEUMONIA_PATH = "model/model_pneumonia.pth"
MODEL_TYPE_PATH = "model/model_type.pth"

CLASSES_PNEUMONIA = ['NORMAL', 'PNEUMONIA']
CLASSES_TYPE = ['none', 'radio']  # mesmo nome das pastas

# Transformação igual à usada na validação
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# Função para carregar modelos
# ─────────────────────────────────────────────
def carregar_modelo(path, num_classes):
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(model._fc.in_features, num_classes)
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Carrega ambos os modelos
model_pneumonia = carregar_modelo(MODEL_PNEUMONIA_PATH, len(CLASSES_PNEUMONIA))
model_type = carregar_modelo(MODEL_TYPE_PATH, len(CLASSES_TYPE))


# ─────────────────────────────────────────────
# /predict — Diagnóstico com verificação de tipo
# ─────────────────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas imagens são permitidas.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao abrir a imagem.")

    img_tensor = transform_val(image).unsqueeze(0)

    # ─── 1. Verificar tipo da imagem ───
    with torch.no_grad():
        logits_type = model_type(img_tensor)
        probas_type = torch.softmax(logits_type, dim=1)
        type_idx = probas_type.argmax().item()
        type_conf = probas_type[0, type_idx].item()
        tipo_imagem = CLASSES_TYPE[type_idx]

    if tipo_imagem == "none":
        return JSONResponse({
            "tipo_imagem": "Não é uma radiografia",
            "confianca_tipo": round(type_conf, 4),
            "diagnostico": None,
            "tem_pneumonia": None,
            "classe_predita": None,
            "confianca_pred": None
        })

    # ─── 2. Diagnóstico de pneumonia ───
    with torch.no_grad():
        logits = model_pneumonia(img_tensor)
        probas = torch.softmax(logits, dim=1)
        top_idx = probas.argmax().item()
        confidence = probas[0, top_idx].item()

    pred_class = CLASSES_PNEUMONIA[top_idx]
    tem_pneumonia = pred_class == "PNEUMONIA"

    return JSONResponse({
        "tipo_imagem": "Radiografia",
        "confianca_tipo": round(type_conf, 4),
        "diagnostico": "Pneumonia" if tem_pneumonia else "Pulmão saudável",
        "tem_pneumonia": tem_pneumonia,
        "classe_predita": pred_class,
        "confianca_pred": round(confidence, 4)
    })


# ─────────────────────────────────────────────
# /predict/probas — Ver todas as probabilidades
# ─────────────────────────────────────────────
@app.post("/predict/probas")
async def predict_probas(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas imagens são permitidas.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao abrir a imagem.")

    img_tensor = transform_val(image).unsqueeze(0)

    with torch.no_grad():
        # Probabilidades do tipo de imagem
        logits_type = model_type(img_tensor)
        probas_type = torch.softmax(logits_type, dim=1)[0]
        prob_dict_type = {CLASSES_TYPE[i]: round(float(p), 4) for i, p in enumerate(probas_type)}

        result = {"prob_tipo": prob_dict_type}

        # Se for radiografia, mostra as probabilidades do modelo de pneumonia
        if CLASSES_TYPE[probas_type.argmax().item()] == "radio":
            logits_pneumonia = model_pneumonia(img_tensor)
            probas_pneu = torch.softmax(logits_pneumonia, dim=1)[0]
            prob_dict_pneu = {CLASSES_PNEUMONIA[i]: round(float(p), 4) for i, p in enumerate(probas_pneu)}
            result["prob_pneumonia"] = prob_dict_pneu

    return JSONResponse(result)


# ─────────────────────────────────────────────
# /health — Verificação da API
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok", "message": "API de pneumonia online"})
