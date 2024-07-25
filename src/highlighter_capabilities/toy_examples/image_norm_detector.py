from datetime import datetime
from typing import Dict, Tuple
from uuid import uuid4

import numpy as np
from PIL.Image import Image as PILImage

from highlighter import LabeledUUID
from highlighter.agent.capabilities import Capability, Entity, StreamEvent
from highlighter.client.base_models import EAVT

__all__ = ["ImageStatsDetector"]

IMGAE_MEAN_ATTRIBUTE_UUID = LabeledUUID(int=2, label="image_mean")
IMGAE_STD_ATTRIBUTE_UUID = LabeledUUID(int=3, label="image_std")


class ImageStatsDetector(Capability):

    def process_frame(self, stream, image: PILImage) -> Tuple[StreamEvent, Dict]:
        np_image = np.array(image)

        mean_r = np.mean(np_image[:, :, 0])
        mean_g = np.mean(np_image[:, :, 1])
        mean_b = np.mean(np_image[:, :, 2])

        std_r = np.std(np_image[:, :, 0])
        std_g = np.std(np_image[:, :, 1])
        std_b = np.std(np_image[:, :, 2])

        mean_entity_id = uuid4()
        mean_entity = Entity(
            id=mean_entity_id,
            global_observations=[
                EAVT.make_scalar_eavt(
                    entity_id=mean_entity_id,
                    value=(mean_r, mean_g, mean_b),
                    attribute_id=IMGAE_MEAN_ATTRIBUTE_UUID,
                    time=datetime.now(),
                    pipeline_element_name=self.__class__.__name__,
                )
            ],
        )

        std_entity_id = uuid4()
        std_entity = Entity(
            id=std_entity_id,
            global_observations=[
                EAVT.make_scalar_eavt(
                    entity_id=std_entity_id,
                    value=(std_r, std_g, std_b),
                    attribute_id=IMGAE_STD_ATTRIBUTE_UUID,
                    time=datetime.now(),
                    pipeline_element_name=self.__class__.__name__,
                )
            ],
        )

        return StreamEvent.OKAY, {"entities": {mean_entity_id: mean_entity, std_entity_id: std_entity}}
