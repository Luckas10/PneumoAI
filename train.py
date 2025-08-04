import os, math, random, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

SEED = 42
random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────
# 1 ▪ Transformações de imagem
# ─────────────────────────────────────────────────────────────
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────
# 2 ▪ Função de treinamento
# ─────────────────────────────────────────────────────────────
def _treinar_modelo(data_dir, batch, epochs, pth_out, tag):
    train_dir = os.path.join(data_dir, "train")
    val_dir   = os.path.join(data_dir, "val")

    ds_tr = datasets.ImageFolder(train_dir, transform=transform_train)
    ds_va = datasets.ImageFolder(val_dir,   transform=transform_val)
    num_cls = len(ds_tr.classes)

    # Balanceamento por classe
    counts = torch.bincount(torch.tensor(ds_tr.targets), minlength=num_cls).float()
    weights_cls = 1. / counts
    weights_per_sample = weights_cls[ds_tr.targets]

    sampler = WeightedRandomSampler(weights_per_sample, len(ds_tr), replacement=True)
    loader_tr = DataLoader(ds_tr, batch_size=batch, sampler=sampler, num_workers=4, pin_memory=True)
    loader_va = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    # Modelo
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_cls)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Otimizador e perda
    loss_fn = nn.CrossEntropyLoss(weight=weights_cls.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    iters_per_ep = len(loader_tr)
    warm_iters = 5 * iters_per_ep
    total_iters = epochs * iters_per_ep
    def lr_lambda(i):
        if i < warm_iters:
            return (i + 1) / warm_iters
        prog = (i - warm_iters) / (total_iters - warm_iters)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA
    ema_decay = 0.999
    ema = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    from torch.amp import autocast, GradScaler
    scaler = GradScaler()

    best_acc, stop_counter, patience = 0.0, 0, 8
    print(f"[{tag}] Classes: {ds_tr.classes}")

    for ep in range(1, epochs + 1):
        model.train()
        loss_tr, correct_tr = 0.0, 0

        for imgs, lbls in loader_tr:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(imgs)
                loss = loss_fn(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)

            loss_tr += loss.item() * imgs.size(0)
            correct_tr += (logits.argmax(1) == lbls).sum().item()

        model.eval()
        loss_va, correct_va = 0.0, 0
        with torch.no_grad(), autocast(device_type='cuda'):
            for imgs, lbls in loader_va:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs)
                loss = loss_fn(logits, lbls)
                loss_va += loss.item() * imgs.size(0)
                correct_va += (logits.argmax(1) == lbls).sum().item()

        acc_tr = correct_tr / len(ds_tr)
        acc_va = correct_va / len(ds_va)

        print(f"[{tag}] Ep {ep}/{epochs} | "
              f"L_tr {loss_tr/len(ds_tr):.4f} A_tr {acc_tr:.3f} | "
              f"L_val {loss_va/len(ds_va):.4f} A_val {acc_va:.3f}")

        if acc_va > best_acc:
            best_acc = acc_va
            stop_counter = 0
            torch.save(ema, pth_out)
            print(f"[{tag}] Novo melhor modelo salvo com acc {best_acc:.3f}")
        else:
            stop_counter += 1
            if stop_counter >= patience:
                print(f"[{tag}] Early stopping após {patience} epochs sem melhora.")
                break

    model.load_state_dict(torch.load(pth_out), strict=False)
    return model

# ─────────────────────────────────────────────────────────────
# 3 ▪ Execução do treinamento para pneumonia
# ─────────────────────────────────────────────────────────────
def treinar_modelo_pneumonia(
    data_dir='chest_xray', batch=16, epochs=20,
    pth='model/model_pneumonia.pth'):
    _treinar_modelo(data_dir, batch, epochs, pth, tag='PNEUMONIA')

if __name__ == '__main__':
    treinar_modelo_pneumonia()
