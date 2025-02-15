import hashlib

import cv2
import numpy as np
from utils.color import Color
from utils.ctf import CTF

hash = hashlib.sha512()


ctf = CTF("114514")


@ctf.flag(ja="色を変更", en="Change the color")
def change_color(img: cv2.typing.MatLike):
    mask = np.all(img == Color.black.ndarray, axis=-1)
    random_mask = np.random.rand(*mask.shape) < 0.5
    img[np.logical_and(mask, random_mask)] = Color.red.ndarray
    mask = np.all(img == Color.white.ndarray, axis=-1)
    random_mask = np.random.rand(*mask.shape) < 0.5
    img[np.logical_and(mask, random_mask)] = Color.blue.ndarray
    return img


@ctf.flag(ja="分割", en="Split")
def split(img: cv2.typing.MatLike):
    h, w = img.shape[:2]
    pad = CTF.BOX_SIZE * 2
    canvas = np.full((h + pad, w + pad, 3), 255, dtype=np.uint8)
    canvas[0 : h // 2, 0 : w // 2] = img[0 : h // 2, 0 : w // 2]
    canvas[0 : h // 2, w // 2 + pad : w + pad] = img[0 : h // 2, w // 2 :]
    canvas[h // 2 + pad : h + pad, 0 : w // 2] = img[h // 2 :, 0 : w // 2]
    canvas[h // 2 + pad : h + pad, w // 2 + pad : w + pad] = img[h // 2 :, w // 2 :]
    return canvas


ctf.once("debug")
ctf.run("flag", 64)
