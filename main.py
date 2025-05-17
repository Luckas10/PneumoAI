# Exemplo básico para saber como pegar imagem com o fastapi

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

app = FastAPI()

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    # Verifica se é um tipo de imagem válido
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Apenas imagens são permitidas.")

    # Lê bytes da imagem
    contents = await file.read()
    try:
        # Abre com PIL para, por exemplo, obter dimensões
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Falha ao processar a imagem.")

    width, height = image.size

    # Aqui você poderia executar qualquer lógica: classificação,
    # OCR, redimensionamento, etc.

    return JSONResponse({
        "filename": file.filename,
        "content_type": file.content_type,
        "width": width,
        "height": height,
        "message": "Imagem recebida com sucesso!"
    })
