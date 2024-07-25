from datetime import datetime
from typing import Dict, Tuple
from uuid import uuid4

from PIL.Image import Image as PILImage

from highlighter import LabeledUUID
from highlighter.agent.capabilities import Capability, Entity, StreamEvent
from highlighter.client.base_models import EAVT

__all__ = ["ImageSizeDetector"]

IMGAE_SHAPE_ATTRIBUTE_UUID = LabeledUUID(int=1, label="image_shape")


class ImageSizeDetector(Capability):

    def process_frame(self, stream, image: PILImage) -> Tuple[StreamEvent, Dict]:
        entity_id = uuid4()
        entity = Entity(
            id=entity_id,
            global_observations=[
                EAVT.make_scalar_eavt(
                    entity_id=entity_id,
                    value=image.shape,
                    attribute_id=IMGAE_SHAPE_ATTRIBUTE_UUID,
                    time=datetime.now(),
                    pipeline_element_name=self.__class__.__name__,
                )
            ],
        )
        self.logger.info(f"{entity}")
        return StreamEvent.OKAY, {"entities": {entity_id: entity}}
