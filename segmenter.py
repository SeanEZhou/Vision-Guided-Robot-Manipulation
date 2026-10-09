"""Learned instance segmentation: Mask R-CNN (torchvision, COCO-pretrained) fine-tuned on simulator images
of the source bin (scripts/make_seg_dataset.py, scripts/train_segmenter.py).

The network sees only the RGB image. It replaces the two hand-built stages of object_perception.py,
geometric segmentation and shape-based identification; back-projection and pose fitting are unchanged.
"""
import time

import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

CLASSES = ["background", "milk", "bread", "cereal", "can"]
SCORE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
ROI_FRACTION = 0.5  # a detection must have at least this share of its mask inside the region of interest


def build_model(pretrained=True):
    model = maskrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, len(CLASSES))
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, len(CLASSES))
    return model


class Segmenter:
    """segmenter(rgb, roi=None) -> (masks, labels): boolean (H, W) masks and class names, one per object.
    Each object type appears at most once in this task, so only the highest-scoring detection of each
    class is kept (an assumption to drop for scenes with duplicate items). With `roi`, a boolean (H, W)
    region of interest such as the source bin, detections mostly outside it are discarded first: the
    network also sees objects already placed in the target bin, which must not compete for their class."""

    def __init__(self, weights_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.last_latency_s = 0.0
        self.last_scores = []

    @torch.no_grad()
    def __call__(self, rgb, roi=None):
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float() / 255.0
        out = self.model([x])[0]
        if self.device == "cuda":
            torch.cuda.synchronize()
        best = {}
        roi_t = None if roi is None else torch.from_numpy(np.ascontiguousarray(roi)).to(self.device)
        for score, label, mask in zip(out["scores"], out["labels"], out["masks"]):
            label = int(label)
            if score < SCORE_THRESHOLD or (label in best and score <= best[label][0]):
                continue
            mask = mask[0] > MASK_THRESHOLD
            if roi_t is not None and (mask & roi_t).sum() < ROI_FRACTION * mask.sum():
                continue
            best[label] = (float(score), mask)
        masks = [m.cpu().numpy() for _, m in best.values()]
        labels = [CLASSES[k] for k in best]
        self.last_scores = [s for s, _ in best.values()]
        self.last_latency_s = time.perf_counter() - t0
        return masks, labels
