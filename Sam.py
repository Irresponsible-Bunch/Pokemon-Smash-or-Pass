# sam_helper.py
import io
from os import remove
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

def post_process_mask(mask):
    # Apply morphological closing to remove small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold back to binary
    
    return mask

def refine_with_grabcut(image, mask):
    # Initialize GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Convert mask for GrabCut
    grabcut_mask = np.where((mask == 255), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    
    # Run GrabCut
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    
    # Final mask
    refined_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return refined_mask

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

    best_mask = masks[np.argmax(scores)].astype(np.uint8) * 255
    
    # Post-process the mask
    best_mask = post_process_mask(best_mask)
    
    # Optional: Refine with GrabCut
    best_mask = refine_with_grabcut(image_bgr, best_mask)
    
    if invert:
        best_mask = ~best_mask

    rgba = np.dstack((image, best_mask.astype(np.uint8)))
    return Image.fromarray(rgba)

def post_process_mask(mask):
    # Apply morphological closing to remove small holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 0)
    mask = (mask > 0.5).astype(np.uint8) * 255  # Threshold back to binary
    
    return mask
from rembg import remove
from PIL import Image
import io

def remove_background_grabcut(pil_image, foreground_point=None, invert=False):
    """
    Remove background using only OpenCV GrabCut based on a foreground point.
    """
    image = np.array(pil_image.convert("RGB"))
    h, w = image.shape[:2]

    if foreground_point is None:
        foreground_point = (w // 2, h // 2)

    # Initial mask: probable background everywhere
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Mark a small region around the point as probable foreground
    px, py = foreground_point
    r = 5
    mask[max(0, py - r):min(h, py + r), max(0, px - r):min(w, px + r)] = cv2.GC_PR_FGD

    # Init models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Run GrabCut
    cv2.grabCut(image, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Final mask
    result_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Post-process for smooth edges
    result_mask = post_process_mask(result_mask)

    if invert:
        result_mask = ~result_mask

    rgba = np.dstack((image, result_mask))
    return Image.fromarray(rgba)


def remove_background_rembg(pil_image, post_process=True):
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    try:
        # Remove background
        result_bytes = remove(img_byte_arr, post_process=post_process)
        
        # Convert back to PIL Image
        result_image = Image.open(io.BytesIO(result_bytes))
        return result_image
    except Exception as e:
        print(f"Error in background removal: {e}")
        return pil_image  # Return original if error occurs
    
    