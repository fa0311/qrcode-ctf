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
from utils.model import FlagDescription, FlagModelMulti, FlagModelOne


class QRCode:
    BOX_SIZE = 10

    def __init__(self, img: cv2.typing.MatLike, text: str):
        self.img = img
        self.text = text

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


FlagCallable = Callable[[cv2.typing.MatLike], list[cv2.typing.MatLike]]


class CTF:
    flags: list[tuple[str, FlagCallable, FlagDescription]] = []
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
            desc = FlagDescription(ja=ja, en=en)
            self.flags.append((func.__name__, func, desc))
            return func

        return wrapper

    def get_key(self):
        key = hashlib.md5(str(random.random()).encode()).hexdigest()
        return "FLAG_{" + key + "}"

    def add_border(self, imgs: list[cv2.typing.MatLike]):
        results: list[cv2.typing.MatLike] = []
        for img in imgs:
            img = cv2.copyMakeBorder(
                img,
                self.BOX_SIZE,
                self.BOX_SIZE,
                self.BOX_SIZE,
                self.BOX_SIZE,
                cv2.BORDER_CONSTANT,
                value=(0, 255, 0),
            )
            results.append(img)
        return results

    def reduce_resolution(self, img: cv2.typing.MatLike):
        h, w = img.shape[:2]
        img = cv2.resize(img, (w // self.BOX_SIZE, h // self.BOX_SIZE))
        return img

    def one(self, dir: str) -> list[FlagModelOne]:
        keys: dict[str, str] = {}
        for name, func, _ in self.flags:
            os.makedirs(f"{dir}/{name}", exist_ok=True)
            self.set_seed(name)
            keys[name] = self.get_key()
            qr = QRCode.from_text(keys[name])
            cv2.imwrite(f"{dir}/{name}/debug.png", qr.img)
            img = func(qr.img.copy())
            for i, img in enumerate(img):
                cv2.imwrite(f"{dir}/{name}/{i}.png", img)
                reduced = self.reduce_resolution(img)
                cv2.imwrite(f"{dir}/{name}/{i}_reduced.png", reduced)
        return [
            FlagModelOne(
                name=name,
                description=desc,
                key=keys[name],
            )
            for name, _, desc in self.flags
        ]

    def grid(self, images: list[cv2.typing.MatLike]):
        cnt = math.ceil(math.sqrt(len(images)))
        col, row = np.ndarray([]), np.ndarray([])
        for i, img in enumerate(images):
            row = img if i % cnt == 0 else cv2.hconcat([row, img])
            if i % cnt == cnt - 1:
                col = row if i == cnt - 1 else cv2.vconcat([col, row])
        return col

    def multi(self, dir: str, count: int) -> list[FlagModelMulti]:
        os.makedirs(dir, exist_ok=True)
        keys: dict[str, list[str]] = {}
        for name, func, _ in self.flags:
            os.makedirs(f"{dir}/{name}", exist_ok=True)
            self.set_seed(name)
            keys[name] = []
            results: list[list[cv2.typing.MatLike]] = []
            debug: list[cv2.typing.MatLike] = []
            for _ in range(count):
                keys[name].append(self.get_key())
                qr = QRCode.from_text(keys[name][-1])
                img = func(qr.img.copy())
                img = self.add_border(img)
                qr_img = self.add_border([qr.img])[0]
                results.append(img)
                debug.append(qr_img)

            for i, imgs in enumerate(zip(*results)):
                cv2.imwrite(f"{dir}/{name}/{i}.png", self.grid(list(imgs)))
            cv2.imwrite(f"{dir}/{name}/debug.png", self.grid(debug))

        return [
            FlagModelMulti(
                name=name,
                description=desc,
                key=keys[name],
            )
            for name, _, desc in self.flags
        ]
