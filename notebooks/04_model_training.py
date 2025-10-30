import sys
import os
print(os.getcwd())
from dataloader import CocoDetDataset, collate_fn

ROOT = "../dataset"  # passe an: Basisordner, der images/ und annotations/ enthält

train_ds = CocoDetDataset(f"{ROOT}/annotations/instances_train.json", f"{ROOT}/images/train", resize_short=640)
val_ds   = CocoDetDataset(f"{ROOT}/annotations/instances_val.json",   f"{ROOT}/images/val",   resize_short=640)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=4, collate_fn=collate_fn, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True)

num_classes_fg = len(train_ds.cid_to_lbl)  # Anzahl Vordergrundklassen

import torch
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)

in_feats = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_feats, num_classes_fg + 1)
model = model.to(device)

from torch.optim import AdamW
from tqdm.auto import tqdm
import time

# Adam-Optimierers mit Weight Decay
# Fügt einen kleinen Strafterm hinzu, der große Gewichtswerte im Modell bestraft
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# Mixed Precision Training, normalerweise 32-Bit Fließkommazahlen (float32) verwendet, aber hier teilweise 16-Bit (float16)
# Spart Speicher und kann die Geschwindigkeit erhöhen, besonders auf neueren GPUs
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

EPOCHS = 40
model.train()

loss_per_step = []
loss_per_epoch = []

for epoch in range(EPOCHS):
    total = 0.0
    n_steps = 0
    start = time.time()
    # enumerate() liefert Index mit
    # tqdm() zeigt Fortschrittsbalken an
    for i, (imgs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        # Alle Bilder im Batch auf GPU schieben
        imgs = [im.to(device) for im in imgs]
        # Alle Targets im Batch auf GPU schieben
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]
        # Gradienten aller Modellparameter auf null zurücksetzen
        optimizer.zero_grad(set_to_none=True)
        # Automatic Mixed Precision (AMP) Modus
        # autocast() entscheidet automatisch, welche Operationen in float16 und welche in float32
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            # Vorwärtsdurchlauf: Modell gibt ein Dictionary mit verschiedenen Verlusten zurück
            loss_dict = model(imgs, targets)
            # Gesamten Verlust berechnen
            loss = sum(loss_dict.values())
        # Backward-Pass mit skaliertem Verlust
        scaler.scale(loss).backward()
        # Modellparameter aktualisieren
        scaler.step(optimizer)
        # Skalierungsfaktor aktualisieren, damit kleine Gradienten nicht zu klein werden
        scaler.update()
        total += loss.item()
        n_steps += 1
        loss_per_step.append(loss.item())
    
    epoch_loss = total / n_steps
    loss_per_epoch.append(epoch_loss)
    dur = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} - train loss: {epoch_loss:.4f} ({dur:.1f}s)")

    torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}.pth")

plt.figure()
plt.plot(loss_per_step)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training loss per step")
plt.savefig("metrics/loss_per_step.png")
plt.show()

plt.figure()
plt.plot(loss_per_epoch, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training loss per epoch")
plt.savefig("metrics/loss_per_epoch.png")
plt.show()

