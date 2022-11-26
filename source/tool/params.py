from pprint import pformat
from typing import Union

import json5
from torch import NumberType


class _Train:
    def __init__(
        self,
        device: str,
        dtype: str,
        data_start: int,
        data_stop: int,
        batch_size: int,
        epochs: int,
        grad_clip_norm: Union[None, NumberType],
        learning_rate: NumberType,
        save_per_epoch: int,
        max_time_length: int,
        seed: Union[None, int] = None,
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")

        self.data_start = data_start
        self.data_stop = data_stop
        self.seed = seed
        self.batch_size = batch_size
        self.epochs = epochs
        self.grad_clip_norm = grad_clip_norm
        self.learning_rate = learning_rate
        self.save_per_epoch = save_per_epoch
        self.device = device
        self.dtype = dtype
        self.max_time_length = max_time_length

    @property
    def kwargs(self):
        return self.__dict__


class _Eval:
    def __init__(
        self, device: str, dtype: str, data_start: int, data_stop: int, training: bool
    ) -> None:
        assert device in ("cpu", "cuda")
        assert dtype in ("float16", "float32")

        self.device = device
        self.dtype = dtype
        self.data_start = data_start
        self.data_stop = data_stop
        self.training = training

    @property
    def kwargs(self):
        return self.__dict__


class Params:
    def __init__(self, path) -> None:
        self.raw_ = json5.load(open(path))

        self.model: str = self.raw_["model"]
        self.train = _Train(**self.raw_["train"])

    def __str__(self):
        return pformat(self.raw_)


class ParamsEval(_Eval):
    def __init__(self, path) -> None:
        self.raw_ = json5.load(open(path))
        super().__init__(**self.raw_["eval"])

    @property
    def kwargs(self):
        ret = self.__dict__.copy()
        ret.pop("_raw")
        return ret

    def __str__(self):
        return pformat(self.raw_)
