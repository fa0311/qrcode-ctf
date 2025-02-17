import json
import shutil

import cv2
import numpy as np
from utils.color import Color
from utils.ctf import CTF
from utils.model import Env

env = Env()
ctf = CTF(env.ctf_seed)


@ctf.flag(ja="アスペクト比が変更された画像", en="Image with changed aspect ratio")
def aspect_ratio(img: cv2.typing.MatLike):
    ratio = 5
    h, w = img.shape[:2]
    canvas = np.full_like(img, 255)
    img = cv2.resize(img, (w, h // ratio))
    canvas[(h - h // ratio) // 2 : (h + h // ratio) // 2] = img
    return [canvas]


@ctf.flag(ja="赤と青に色が変わった画像", en="Image with red and blue colors changed")
def replace_color(img: cv2.typing.MatLike):
    mask = np.all(img == Color.black.ndarray, axis=-1)
    random_mask = np.random.rand(*mask.shape) < 0.5
    img[np.logical_and(mask, random_mask)] = Color.red.ndarray
    mask = np.all(img == Color.white.ndarray, axis=-1)
    random_mask = np.random.rand(*mask.shape) < 0.5
    img[np.logical_and(mask, random_mask)] = Color.blue.ndarray
    return [img]


@ctf.flag(ja="分割され間に余白がある画像", en="Image divided with padding in between")
def split(img: cv2.typing.MatLike):
    h, w = img.shape[:2]
    pad = CTF.BOX_SIZE * 2
    canvas = np.full((h + pad, w + pad, 3), Color.white.ndarray)
    canvas[0 : h // 2, 0 : w // 2] = img[0 : h // 2, 0 : w // 2]
    canvas[0 : h // 2, w // 2 + pad : w + pad] = img[0 : h // 2, w // 2 :]
    canvas[h // 2 + pad : h + pad, 0 : w // 2] = img[h // 2 :, 0 : w // 2]
    canvas[h // 2 + pad : h + pad, w // 2 + pad : w + pad] = img[h // 2 :, w // 2 :]

    return [canvas]


@ctf.flag(ja="輪郭線が描かれた画像", en="Image with contour lines")
def outline(img: cv2.typing.MatLike):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros_like(img)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1:]
    contour = cv2.drawContours(canvas, contours, -1, (255, 255, 255), 1)
    contour = cv2.bitwise_not(contour)
    return [contour]


@ctf.flag(ja="赤い点が追加された画像", en="Image with red dots added")
def dot_noise(img: cv2.typing.MatLike):
    mask = np.zeros(img.shape[:2], dtype=bool)
    index = CTF.BOX_SIZE // 2 - 1
    size = CTF.BOX_SIZE // 2 + 3
    mask[index :: CTF.BOX_SIZE, index :: CTF.BOX_SIZE] = True
    mask = cv2.dilate(
        mask.astype(np.uint8), np.ones((size, size), np.uint8), iterations=1
    )
    img[mask.astype(bool)] = Color.red.ndarray
    return [img]


@ctf.flag(ja="QRコードが分割された画像", en="Image with QR code divided")
def or_split(img: cv2.typing.MatLike):
    h, w = img.shape[:2]
    sh, sw = h // CTF.BOX_SIZE, w // CTF.BOX_SIZE
    mask = np.random.choice([0, 255], (sh, sw), p=[0.5, 0.5]).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    img1 = np.full_like(img, 255)
    img2 = np.full_like(img, 255)
    img1[mask == 0] = img[mask == 0]
    img2[mask == 255] = img[mask == 255]
    return [img1, img2]


@ctf.flag(ja="ドット柄", en="Dot pattern")
def dot_pattern(img: cv2.typing.MatLike):
    mask1 = np.zeros(img.shape[:2], dtype=bool)
    mask1[CTF.BOX_SIZE // 2 :: CTF.BOX_SIZE, CTF.BOX_SIZE // 2 :: CTF.BOX_SIZE] = True
    mask2 = img[:, :, 0] == 0
    and_mask = np.logical_and(mask1, mask2)
    canvas = np.full_like(img, 255)
    canvas[and_mask] = Color.black.ndarray
    return [canvas]


shutil.rmtree("flags", ignore_errors=True)

one = ctf.one("flags/one")
multi = ctf.multi("flags/multi", 64)

with open("flags/one/data.json", "w") as f:
    json.dump([x.model_dump(mode="json") for x in one], f)
with open("flags/multi/data.json", "w") as f:
    json.dump([x.model_dump(mode="json") for x in multi], f)
