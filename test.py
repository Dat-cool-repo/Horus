import cv2
import numpy as np
import os

# Install with: pip install paddlepaddle-gpu paddleocr
# or for CPU-only (MKL-enabled): pip install paddlepaddle paddleocr -f ...
from paddleocr import PaddleOCR, draw_ocr

from PIL import ImageFont
import matplotlib.pyplot as plt

# Optional: for dictionary-based text correction (see post_process_text function)
try:
    from spellchecker import SpellChecker
    spell = SpellChecker()
except ImportError:
    spell = None

# --------------------------------------------------
# 1. Initialize PaddleOCR with Accuracy Focus
# --------------------------------------------------
# - use_angle_cls=True helps correct text that is rotated 180°, 90°, etc.
# - det_db_thresh, det_db_box_thresh can help capture faint text
# - Use the latest and/or 'server' model versions for better accuracy if needed
#
#    e.g., en_PP-OCRv4_server_infer  or  ch_PP-OCRv4_server_infer
#    Check https://github.com/PaddlePaddle/PaddleOCR for the latest models.
#
# Adjust model_dir paths to your local paths accordingly.
# --------------------------------------------------
ocr = PaddleOCR(
    det_model_dir="path/to/en_PP-OCRv4_det_infer",  # or en_PP-OCRv4_server_infer for higher accuracy
    rec_model_dir="path/to/en_PP-OCRv4_rec_infer",
    use_angle_cls=True,
    lang='en',
    det_db_thresh=0.3,       # Adjust if text is missed / or too many false boxes
    det_db_box_thresh=0.5    # Filter out boxes with low confidence
)

# --------------------------------------------------
# 2. Image Quality Improvement
# --------------------------------------------------
def improve_image_quality(image):
    """
    Optional advanced pre-processing to improve OCR accuracy:
     - Convert to grayscale
     - Denoise
     - Adjust contrast / brightness
     - (Optionally) apply adaptive thresholding for high-contrast text
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1) Denoise (fastNlMeansDenoising is good for mild noise)
    #    h parameter can be tuned (strength of the filter).
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 2) Histogram equalization (improves global contrast)
    equalized = cv2.equalizeHist(denoised)

    # 3) (Optionally) Adaptive Threshold
    #    This can help if text is dark on a bright, non-uniform background.
    #    If your images are already high-quality, you might skip thresholding.
    #    Comment out if it's causing any issues.
    thresh = cv2.adaptiveThreshold(
        equalized, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Convert single-channel back to 3-channel if needed
    # because later code might expect BGR
    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return processed

# --------------------------------------------------
# 3. Deskew using Hough Lines
# --------------------------------------------------
def deskew_with_hough(image):
    """
    Deskew the input image by detecting predominant lines via Hough transform.
    Returns a rotated (deskewed) version of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        # Fallback to the original image if no lines are found
        return image

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert angle to degrees
        angle_deg = np.degrees(theta)
        # Adjust angles to align vertical lines to 0 degrees
        angles.append(angle_deg - 90)

    # Use the median angle to reduce outlier influence
    median_angle = np.median(angles)

    # Rotate the image by the negative of this angle to deskew
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed

# --------------------------------------------------
# 4. Detect and Warp Document
# --------------------------------------------------
def detect_and_warp_document(deskewed):
    """
    Finds the largest contour in the deskewed image and attempts
    to warp it as if it's a document. If no 4-point contour is found,
    returns the deskewed image as a fallback.
    """
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return deskewed  # Fallback

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Approximate polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Warp perspective if we have 4 corners
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)

        rect[0] = pts[np.argmin(s)]   # Top-left
        rect[2] = pts[np.argmax(s)]   # Bottom-right
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left

        (tl, tr, br, bl) = rect

        # Compute dimensions of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        # Destination coordinates
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Perspective transform
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(deskewed, M, (maxWidth, maxHeight))
        return warped
    else:
        return deskewed  # Fallback if no 4-corner contour

# --------------------------------------------------
# 5. (Optional) Post-process text with Spell Checker
# --------------------------------------------------
def post_process_text_line(text_line):
    """
    Example function using pyspellchecker to correct English words.
    Only works if the text is purely alphabetical and in dictionary.
    This is a simple example and may need domain-specific logic.
    """
    if not spell:
        return text_line  # Spellchecker not installed

    # Split line into words
    words = text_line.split()

    corrected_words = []
    for w in words:
        # Spellchecker correction
        corrected = spell.correction(w)
        corrected_words.append(corrected)

    return " ".join(corrected_words)

def post_process_all_text(lines):
    """
    Applies spellchecker-based post-processing to each recognized line.
    """
    return [post_process_text_line(line) for line in lines]

# --------------------------------------------------
# 6. Main Preprocessing Function
# --------------------------------------------------
def preprocess_image(image_path):
    """
    1. Read the image
    2. Improve image quality (denoise, threshold, etc.)
    3. Deskew using Hough lines
    4. Detect & warp the document region
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {image_path}")

    # Step A: Improve image quality
    processed = improve_image_quality(image)

    # Step B: Deskew
    deskewed = deskew_with_hough(processed)

    # Step C: Warp perspective (destretch)
    warped = detect_and_warp_document(deskewed)

    return warped

# --------------------------------------------------
# 7. Run the OCR & Save Results
# --------------------------------------------------
def main():
    image_path = 'test.jpg'   # <-- Update with your image path

    # Preprocess
    preprocessed_image = preprocess_image(image_path)

    # Run OCR
    # Because we already used pre-processing steps, pass the final image array directly.
    # Note: ocr.ocr() can take either a file path or a NumPy array.
    results = ocr.ocr(preprocessed_image, det=True, rec=True, cls=True)

    # Extract the recognized text lines
    raw_text_lines = [line[1][0] for line in results[0]]

    # Optional post-processing (spell check, etc.)
    corrected_text_lines = post_process_all_text(raw_text_lines)

    # Join as final output text
    output_text = "\n".join(corrected_text_lines)

    # Save text to a file with the same name as the image
    output_file = os.path.splitext(image_path)[0] + ".txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"OCR results saved to {output_file}")

    # Extract bounding boxes, texts, and scores for visualization
    boxes = [line[0] for line in results[0]]
    texts = corrected_text_lines  # After post-processing
    scores = [line[1][1] for line in results[0]]

    # Specify a font path; update for your system if needed
    try:
        font_path = r"C:\\Windows\\Fonts\\arial.ttf"  # Update path to a valid .ttf
        font = ImageFont.truetype(font_path, size=20)
    except OSError:
        print("Font file not found. Using a default font.")
        font_path = None

    # Draw OCR results on the image
    annotated_image = draw_ocr(preprocessed_image, boxes, texts, scores, font_path=font_path)

    # Convert from BGR to RGB for matplotlib display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Show the resulting image
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title("OCR Result")
    plt.show()

if __name__ == "__main__":
    main()
