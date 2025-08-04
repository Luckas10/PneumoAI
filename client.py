import requests

API_URL = "http://localhost:8000"  # Altere se estiver usando outra porta ou hospedagem

# Caminho da imagem para envio
caminho_imagem = input("Digite o caminho da imagem: ")

def enviar_para_predict():
    try:
        with open(caminho_imagem, "rb") as f:
            files = {"file": (caminho_imagem, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files)
        print("Diagnóstico:")
        print(response.json())
    except:
        print("Erro na busca do arquivo!")

def enviar_para_probas():
    with open(caminho_imagem, "rb") as f:
        files = {"file": (caminho_imagem, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict/probas", files=files)
    print("Probabilidades:")
    print(response.json())


def verificar_status_api():
    response = requests.get(f"{API_URL}/health")
    print("Status da API:")
    print(response.json())


if __name__ == "__main__":
    print("1. Verificando API...")
    verificar_status_api()

    print("\n2. Realizando diagnóstico...")
    enviar_para_predict()

    print("\n3. Checando probabilidades...")
    enviar_para_probas()
