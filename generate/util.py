import hashlib
import math
import random
from typing import Callable

import cv2
import numpy as np
import qrcode
import qrcode.constants
import qrcode.image.pil


class Square:
    img: cv2.typing.MatLike

    def __init__(self, img: cv2.typing.MatLike):
        self.check_square(img)
        self.img = img

    def check_square(self, img: cv2.typing.MatLike):
        if img.shape[0] != img.shape[1]:
            raise ValueError("Image is not square")

    def size(self) -> int:
        return self.img.shape[0]

    def resize(self, size: int):
        self.img = cv2.resize(self.img, (size, size))
        return self

    def save(self, path: str):
        cv2.imwrite(path, self.img)
        return self

    def run(self, callback: Callable[[cv2.typing.MatLike], cv2.typing.MatLike]):
        img = callback(self.img)
        self.check_square(img)
        self.img = img
        return self


class QRCode(Square):
    BOX_SIZE = 10

    def __init__(self, img: cv2.typing.MatLike, text: str):
        super().__init__(img)
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
        return cls(image, text)

    def get_box_count(self) -> int:
        return self.size() // self.BOX_SIZE

    def paste_img(self, logo: Square, ratio: float):
        box_count = round(self.get_box_count() / ratio)
        box_count = box_count if self.get_box_count() % 2 == 0 else box_count + 1
        logo = logo.resize(self.BOX_SIZE * box_count)
        paste = round((self.size() - logo.size()) / 2)
        self.img[paste : paste + logo.size(), paste : paste + logo.size()] = logo.img
        return self

    def check(self):
        detector = cv2.QRCodeDetector()
        data, bbox, rectifiedImage = detector.detectAndDecode(self.img)
        if data != self.text:
            raise ValueError("QR Code is not correct")


class CTF:
    flags: list[tuple[str, Callable[[cv2.typing.MatLike], cv2.typing.MatLike]]] = []

    def flag(self, name):
        def wrapper(func: Callable[[cv2.typing.MatLike], cv2.typing.MatLike]):
            self.flags.append((name, func))
            return func

        return wrapper

    def get_key(self):
        key = hashlib.md5(str(random.random()).encode()).hexdigest()
        return "FLAG_{" + key + "}"

    def once(self):
        for name, func in self.flags:
            key = self.get_key()
            img = QRCode.from_text(key)
            img.check()
            img = img.run(func)
            img.save(f"output/{name}.png")
            print(key)

    def run(self, count: int):
        sq = math.sqrt(count)
        row_count, col_count = math.ceil(sq), math.floor(sq)
        key = self.get_key()
        img = QRCode.from_text(key)

        for name, func in self.flags:
            answer = []
            col_canvas = None
            for _ in range(col_count):
                row_canvas = None
                for _ in range(row_count):
                    key = self.get_key()
                    img = QRCode.from_text(key)
                    img.check()
                    img = img.run(func)
                    col_canvas = (
                        img.img
                        if row_canvas is None
                        else cv2.hconcat([row_canvas, img.img])
                    )
                    answer.append(key)

                assert col_canvas is not None
                row_canvas = (
                    col_canvas
                    if row_canvas is None
                    else cv2.vconcat([row_canvas, col_canvas])
                )

            cv2.imwrite(f"output/{name}.png", row_canvas)

            print("".join([hashlib.md5(key.encode()).hexdigest()[0] for key in answer]))
