import hashlib
import random

import cv2
from util import CTF

hash = hashlib.sha512()


ctf = CTF()


@ctf.flag("lv1")
def lv1(img: cv2.typing.MatLike):
    h, w = img.shape[:2]
    img1 = img[: h // 2, :]
    img2 = img[h // 2 :, :]
    img = cv2.vconcat([img2, img1])
    return img


random.seed("114514")
ctf.run(64)
