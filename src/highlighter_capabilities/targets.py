import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from highlighter.agent.capabilities import Capability, ContextPipelineElement, Entities, StreamEvent

__all__ = ["EntityWrite"]


class EntityWrite(Capability):
    """
    {
      "name": "EntityWrite",
      "parameters": {
        "output_per_frame": false
        "output_file_path": ""
      },
      "input": [
        {
          "name": "entities",
          "type": "dict"
        },
        {
          "name": "image",
          "type": "Image.Image"
        }
      ],
      "output": [],
      "deploy": {
        "local": {
          "module": "highlighter.agent.capabilities"
        }
      }
    }
    """

    class DefaultStreamParameters(Capability.DefaultStreamParameters):
        # if True: write entities each time process_frame is called
        # if False: compile entities and write all to a singel file when stop_stream is called
        output_per_frame: bool = False

        # can use substitution for output_file_path
        #   - frame_id
        #   - source_file_name -> equivalent to Path(source_file_location).stem
        #
        # If slashes are use the directories will be created.
        #  eg: output_file_path = "my_output/frame_{frame_id}_{source_file_name}.json
        output_file_path: str = ""

    @property
    def output_per_frame(self) -> bool:
        value, _ = self._get_parameter("output_per_frame")
        return value

    @property
    def output_file_path(self) -> str:
        value, _ = self._get_parameter("output_file_path")
        return value

    @classmethod
    def get_capability_input_output_deploy(cls):
        return {
            "input": [
                {"name": "entities", "type": "Entities"},
                {"name": "image", "type": "image"},
            ],
            "output": [{}],
            "deploy": {"local": {"module": "highlighter.agent.capabilities"}},
        }

    def __init__(self, context: ContextPipelineElement):
        super().__init__(context)
        self.entities = defaultdict(list)

    def start_stream(self, stream, stream_id) -> Tuple[StreamEvent, Optional[str]]:
        super().start_stream(stream, stream_id)
        return StreamEvent.OKAY, None

    def process_frame(self, stream, entities, image) -> Tuple[StreamEvent, Entities]:

        # ToDo: Find a better palce to put/get this from
        #       see also, ImageDataSource.process_frame
        source_info = stream.get("source_info", {})
        source_file_location = source_info.get("source_file_location", None)
        source_file_name = Path(source_file_location).stem

        serializable_entities = {
            "frame_id": stream["frame_id"],
            "source_file_location": source_file_location,
            **self._dump_frame_entities(entities),
        }
        if self.output_per_frame:
            # Dump to std out
            json_entites = json.dumps(serializable_entities, indent=2, sort_keys=True)
            print(json_entites, file=sys.stdout)

            if self.output_file_path:
                # Dump to file
                frame_id = stream["frame_id"]
                output_file_path = self.output_file_path.format(
                    frame_id=frame_id,
                    source_file_name=source_file_name,
                )
                Path(output_file_path).parent.mkdir(exist_ok=True, parents=True)
                with open(output_file_path, "w") as f:
                    f.write(json_entites)
        else:
            self.entities[stream["stream_id"]].append(serializable_entities)

        return StreamEvent.OKAY, {}

    def _dump_frame_entities(self, frame_entities: Dict):
        serializable_entities = {}
        for ent_id, entity in frame_entities.items():
            serializable_entities[str(ent_id)] = entity.to_serializable_dict()
        return serializable_entities

    def stop_stream(self, stream, stream_id):
        entities_json = json.dumps(self.entities[stream_id], indent=2, sort_keys=True)

        if not self.output_per_frame:

            if self.output_file_path:
                with open(self.output_file_path, "w") as f:
                    f.write(entities_json)
            else:
                print(entities_json, file=sys.stdout)
