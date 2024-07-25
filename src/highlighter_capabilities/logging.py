import sys
from typing import Dict, Tuple

from highlighter import LabeledUUID
from highlighter.agent.capabilities import Capability, StreamEvent

__all__ = [
    "EchoStreamParamsA",
    "EchoStreamParamsB",
]

STREAM_PARAMS_A_ATTRIBUTE_UUID = LabeledUUID(int=6, label="stream_params_a")
STREAM_PARAMS_B_ATTRIBUTE_UUID = LabeledUUID(int=7, label="stream_params_b")


class EchoStreamParamsA(Capability):

    class DefaultStreamParameters(Capability.DefaultStreamParameters):
        x: str = "SET_IN_CODE"

    @property
    def x(self):
        return self._get_parameter("x")[0]

    def process_frame(self, stream, *args, **kwargs) -> Tuple[StreamEvent, Dict]:
        if kwargs["foo"]:
            print(f"{self.__class__.__name__}.frame_param.foo: {kwargs['foo']}", file=sys.stdout)

        print(f"{self.__class__.__name__}.x: {self.x}", file=sys.stdout)
        return StreamEvent.OKAY, {}


class EchoStreamParamsB(Capability):

    class DefaultStreamParameters(Capability.DefaultStreamParameters):
        x: str = "SET_IN_CODE"

    @property
    def x(self):
        return self._get_parameter("x")[0]

    def process_frame(self, stream, *args, **kwargs) -> Tuple[StreamEvent, Dict]:
        print(f"{self.__class__.__name__}.x: {self.x}", file=sys.stdout)
        return StreamEvent.OKAY, {}
