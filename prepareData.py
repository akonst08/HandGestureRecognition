import os
import csv
import cv2
import mediapipe as mp
import numpy as np
import random

# Dataset path
dataset_dir = './Dataset'
output_csv = 'model/keypoint_classifier/keypoint.csv'
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Class mapping
classes = ["call","dislike","fist","four","like","mute","ok","one","palm","peace_inverted","peace","rock","stop_inverted","stop","three","three2","two_up_inverted","two_up"]
label_to_id = {label: idx for idx, label in enumerate(classes)}

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Augmentation function with increased rotation
def augment_image(image):
    rows, cols = image.shape[:2]
    angle = random.uniform(-45, 45)  #  Wider rotation range
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    tx = random.uniform(-0.1, 0.1) * cols
    ty = random.uniform(-0.1, 0.1) * rows
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    rotated = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    shifted = cv2.warpAffine(rotated, M_trans, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    if random.random() < 0.5:
        shifted = cv2.flip(shifted, 1)
    return shifted

# Landmark extraction (normalized)
def extract_normalized_landmarks_from_img(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None
    landmarks = result.multi_hand_landmarks[0].landmark
    coords = [[lm.x, lm.y] for lm in landmarks]
    base_x, base_y = coords[0]
    rel_coords = [[x - base_x, y - base_y] for x, y in coords]
    flat = [coord for pt in rel_coords for coord in pt]
    max_val = max([abs(v) for v in flat])
    if max_val > 0:
        flat = [v / max_val for v in flat]
    return flat if len(flat) == 42 else None

# Labels for extra rotations
extra_rotation_labels = ["rock", "three"]

# Process only the specified classes
with open(output_csv, mode='a', newline='') as f:
    writer = csv.writer(f)
    total_saved = 0

    for label in extra_rotation_labels:
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            print(f"⚠ Skipping missing label folder: {label}")
            continue

        class_id = label_to_id[label]
        images = [fn for fn in os.listdir(label_path) if fn.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(images) == 0:
            print(f"⚠ No images found in {label}")
            continue

        subset_size = min(300, len(images))
        selected_images = random.sample(images, subset_size)
        print(f"🔹 Processing {subset_size} images for label '{label}'")

        for filename in selected_images:
            image_path = os.path.join(label_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"✘ Skipped (couldn't read): {label}/{filename}")
                continue

            saved_count = 0
            for i in range(8):  # More variations
                aug_image = augment_image(image)
                features = extract_normalized_landmarks_from_img(aug_image)
                if features:
                    writer.writerow([class_id] + features)
                    saved_count += 1
                    total_saved += 1
                    print(f"✔ Saved augmented: {label}/{filename} aug#{i+1}")
                else:
                    print(f"✘ Skipped augmented: {label}/{filename} aug#{i+1}")
            print(f"🔹 Total saved augmentations for {label}/{filename}: {saved_count}/8")

    print(f" Augmentation and extraction complete!")
    print(f" Total samples written: {total_saved}")
