import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F

class CocoDetDataset(Dataset):
    def __init__(self, ann_file, img_root, resize_short=640):
        self.ann = json.load(open(ann_file, "r"))
        self.img_root = Path(img_root)
        self.images = self.ann["images"]
        self.anns   = self.ann["annotations"]
        self.cats   = self.ann["categories"]
        # index: image_id -> list[ann]
        self.by_img = {}
        for a in self.anns:
            self.by_img.setdefault(a["image_id"], []).append(a)
        # COCO category_id -> 1..K (torchvision erwartet labels >=1)
        cat_ids = sorted([c["id"] for c in self.cats])
        self.cid_to_lbl = {cid: i+1 for i, cid in enumerate(cat_ids)}
        self.resize_short = resize_short

    def __len__(self): return len(self.images)

    @staticmethod
    def _xywh_to_xyxy(xywh):
        x,y,w,h = xywh
        return [x, y, x+w, y+h]

    def __getitem__(self, idx):
        meta = self.images[idx]
        img_path = self.img_root / meta["file_name"]
        img = Image.open(img_path).convert("RGB")

        # Resize kurze Kante (einfach & stabil)
        if self.resize_short:
            w, h = img.size
            s = self.resize_short / min(w, h)
            img = img.resize((int(w*s), int(h*s)), Image.BILINEAR)

        boxes, labels = [], []
        for a in self.by_img.get(meta["id"], []):
            xyxy = self._xywh_to_xyxy(a["bbox"])
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            # gleiche Skalierung auf Boxen anwenden
            if self.resize_short:
                w0, h0 = meta["width"], meta["height"]
                s = min(img.size) / min(w0, h0)  # entspricht oben verwendetem s
                xyxy = [v*s for v in xyxy]
            boxes.append(xyxy)
            labels.append(self.cid_to_lbl[a["category_id"]])

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([meta["id"]], dtype=torch.int64),
        }
        img_t = F.to_tensor(img)  # [0..1], CxHxW
        return img_t, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
