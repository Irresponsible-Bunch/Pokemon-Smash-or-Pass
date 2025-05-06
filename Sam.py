# sam_helper.py
import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# Load SAM model once
sam_checkpoint = r"C:\Users\user\Desktop\sam_vit_h_4b8939.pth" # Make sure this path is correct
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)


def remove_background_sam(pil_image, foreground_point=None, invert=False):
    image = np.array(pil_image)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predictor.set_image(image_bgr)

    h, w, _ = image_bgr.shape

    if foreground_point is None:
        input_point = np.array([[w // 2, h // 2]])  # default center
    else:
        input_point = np.array([foreground_point])

    input_label = np.array([1])  # foreground label

    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    best_mask = masks[np.argmax(scores)]

    if invert:
        best_mask = ~best_mask

    rgba = np.dstack((image, (best_mask * 255).astype(np.uint8)))
    return Image.fromarray(rgba)

