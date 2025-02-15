import hashlib
import math
import os
import random
from typing import Callable

import cv2
import numpy as np
import qrcode
import qrcode.constants
import qrcode.image.pil


class QRCode:
    BOX_SIZE = 10

    def __init__(self, img: cv2.typing.MatLike, text: str):
        self.img = img
        self.text = text

    def copy_with(self, img: cv2.typing.MatLike):
        return self.__class__(img, self.text)

    @classmethod
    def from_text(cls, text: str):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=cls.BOX_SIZE,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)
        image = qr.make_image()
        assert isinstance(image, qrcode.image.pil.PilImage)
        image = np.array(image, dtype=np.uint8)
        image = np.where(image == 0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        ins = cls(image, text)
        return ins

    def check(self) -> bool:
        detector = cv2.QRCodeDetector()
        data, bbox, rectifiedImage = detector.detectAndDecode(self.img)
        return data == self.text


FlagCallable = Callable[[cv2.typing.MatLike], cv2.typing.MatLike]


class CTF:
    flags: list[tuple[str, FlagCallable, tuple[str, str]]] = []
    pepper: str
    BOX_SIZE = QRCode.BOX_SIZE

    def __init__(self, pepper: str):
        self.pepper = pepper

    def set_seed(self, seed: str):
        key = hashlib.md5((self.pepper + seed).encode()).hexdigest()
        random.seed(key)
        np.random.seed(int(key, 16) % (2**32))

    def flag(self, ja: str, en: str):
        def wrapper(func: FlagCallable):
            self.flags.append((func.__name__, func, (ja, en)))
            return func

        return wrapper

    def get_key(self):
        key = hashlib.md5(str(random.random()).encode()).hexdigest()
        return "FLAG_{" + key + "}"

    def once(self, dir: str):
        os.makedirs(dir, exist_ok=True)
        for name, func, _ in self.flags:
            self.set_seed(name)
            key = self.get_key()
            img = QRCode.from_text(key)
            cv2.imwrite(f"{dir}/{name}_qr.png", img.img)
            img = func(img.img)
            cv2.imwrite(f"{dir}/{name}.png", img)

    def grid(self, images: list[cv2.typing.MatLike]):
        cnt = math.ceil(math.sqrt(len(images)))
        col, row = np.ndarray([]), np.ndarray([])
        for i, img in enumerate(images):
            row = img if i % cnt == 0 else cv2.hconcat([row, img])
            if i % cnt == cnt - 1:
                col = row if i == cnt - 1 else cv2.vconcat([col, row])
        return col

    def run(self, dir: str, count: int):
        os.makedirs(dir, exist_ok=True)
        for name, func, _ in self.flags:
            self.set_seed(name)
            keys = []
            result = []
            for _ in range(count):
                keys.append(self.get_key())
                img = QRCode.from_text(keys[-1])
                img = img.copy_with(func(img.img))
                img = cv2.copyMakeBorder(
                    img.img, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 255, 0)
                )
                result.append(img)
            cv2.imwrite(f"{dir}/{name}.png", self.grid(result))
            print("".join([hashlib.md5(key.encode()).hexdigest()[0] for key in keys]))
