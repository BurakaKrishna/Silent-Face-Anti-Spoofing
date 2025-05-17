# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')

SAMPLE_IMAGE_PATH = "./images/sample/"


def crop_to_4_3(image):
    """Crops the input image to a 4:3 aspect ratio centered."""
    height, width, _ = image.shape
    target_ratio = 3 / 4
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Too wide — crop width
        new_width = int(height * target_ratio)
        start_x = (width - new_width) // 2
        cropped_image = image[:, start_x:start_x + new_width]
    elif current_ratio < target_ratio:
        # Too tall — crop height
        new_height = int(width / target_ratio)
        start_y = (height - new_height) // 2
        cropped_image = image[start_y:start_y + new_height, :]
    else:
        # Already 4:3
        cropped_image = image

    return cropped_image


def test(image_name, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)

    if image is None:
        print(f"Failed to load image: {SAMPLE_IMAGE_PATH + image_name}")
        return

    # Auto-crop to 4:3 if not already
    height, width, _ = image.shape
    if width / height != 3 / 4:
        print("Image is not appropriate!!! Cropping to 4:3 aspect ratio.")
        image = crop_to_4_3(image)

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from all models
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        print(f"Image '{image_name}' is Real Face. Score: {value:.2f}.")
        result_text = f"RealFace Score: {value:.2f}"
        color = (255, 0, 0)
    else:
        print(f"Image '{image_name}' is Fake Face. Score: {value:.2f}.")
        result_text = f"FakeFace Score: {value:.2f}"
        color = (0, 0, 255)

    print(f"Prediction cost {test_speed:.2f} s")
    cv2.rectangle(
        image,
        (image_bbox[0], image_bbox[1]),
        (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
        color, 2)
    cv2.putText(
        image,
        result_text,
        (image_bbox[0], image_bbox[1] - 5),
        cv2.FONT_HERSHEY_COMPLEX, 0.5 * image.shape[0] / 1024, color)

    format_ = os.path.splitext(image_name)[-1]
    result_image_name = image_name.replace(format_, "_result" + format_)
    cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
