import json

from click.testing import CliRunner

from highlighter.cli.highlighter import highlighter_group

pipeline_definition = """{
  "version": 0,
  "name":    "p_image_io",
  "runtime": "python",

  "graph": ["(ImageSizeDetector)"],

  "elements": [
    { "name":   "ImageSizeDetector",
      "parameters": {
      },
      "input":  [{"name": "image", "type": "image" }],
      "output": [{ "name": "observations", "type": "observations" }],
      "deploy": {
        "local": { "module": "highlighter_capabilities" }
      }
    }
  ]
}
"""


def test_run_agent(tmpdir, data_files_dir):
    runner = CliRunner()
    test_image_path = f"{data_files_dir}/33659.jpg"
    sample_agent_def = f"{tmpdir}/test_agent.json"
    with open(sample_agent_def, "w") as f:
        f.write(pipeline_definition)

    cli_cmd = f"agent run --data-source ImageDataSource --data-target EntityWrite {sample_agent_def} {test_image_path}"
    result = runner.invoke(highlighter_group, cli_cmd.split(), catch_exceptions=False)
    assert result.exit_code == 0
    entities = json.loads(result.stdout[result.stdout.index("[") :])

    assert len(entities) == 1
    for key, value in entities[0].items():
        if key == "frame_id":
            assert value == "0"
        elif key == "source_file_location":
            assert value.endswith("33659.jpg")
        else:
            observation = value["global_observations"][0]
            assert tuple(observation["value"]) == (572, 1008, 3)


def test_capability_stream_parameters(tmpdir):
    pipeline_definition = """{
      "version": 0,
      "name":    "p_image_io",
      "runtime": "python",
    
      "graph": ["(EchoStreamParamsA EchoStreamParamsB)"],
    
      "elements": [
        { "name":   "EchoStreamParamsA",
          "parameters": {
              "x": "SET_IN_DEF"
          },
          "input":  [
              {"name": "foo", "type": "str"}
              ],
          "output": [],
          "deploy": {
            "local": { "module": "highlighter_capabilities" }
          }
        },
        { "name":   "EchoStreamParamsB",
          "parameters": {
          },
          "input":  [],
          "output": [],
          "deploy": {
            "local": { "module": "highlighter_capabilities" }
          }
        }
      ]
    }
    """

    runner = CliRunner(mix_stderr=False)
    sample_agent_def = f"{tmpdir}/test_agent.json"
    with open(sample_agent_def, "w") as f:
        f.write(pipeline_definition)

    cli_cmd = [
        "agent",
        "run",
        f"{sample_agent_def}",
        '[{"foo": ""}]',
    ]
    result = runner.invoke(highlighter_group, cli_cmd, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "EchoStreamParamsA.x: SET_IN_DEF" in result.stdout
    assert "EchoStreamParamsB.x: SET_IN_CODE" in result.stdout

    cli_cmd = [
        "agent",
        "run",
        "-sp",
        "EchoStreamParamsB.x=SET_IN_CLI_WITH_FULL_KEY",
        f"{sample_agent_def}",
        '[{"foo": ""}]',
    ]
    result = runner.invoke(highlighter_group, cli_cmd, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "EchoStreamParamsA.x: SET_IN_DEF" in result.stdout
    assert "EchoStreamParamsB.x: SET_IN_CLI_WITH_FULL_KEY" in result.stdout

    cli_cmd = [
        "agent",
        "run",
        "-sp",
        "EchoStreamParamsA.x=SET_IN_CLI_WITH_FULL_KEY",
        f"{sample_agent_def}",
        '[{"foo": ""}]',
    ]
    result = runner.invoke(highlighter_group, cli_cmd, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "EchoStreamParamsA.x: SET_IN_CLI_WITH_FULL_KEY" in result.stdout
    assert "EchoStreamParamsB.x: SET_IN_CODE" in result.stdout

    cli_cmd = [
        "agent",
        "run",
        "-sp",
        "EchoStreamParamsA.x=SET_IN_CLI_WITH_FULL_KEY",
        "-sp",
        "x=SET_IN_CLI_WITH_GLOBAL_KEY",
        f"{sample_agent_def}",
        '[{"foo": ""}]',
    ]
    result = runner.invoke(highlighter_group, cli_cmd, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "EchoStreamParamsA.x: SET_IN_CLI_WITH_FULL_KEY" in result.stdout
    assert "EchoStreamParamsB.x: SET_IN_CLI_WITH_GLOBAL_KEY" in result.stdout

    # Directly send frame params
    cli_cmd = [
        "agent",
        "run",
        f"{sample_agent_def}",
        '[{"foo": "FOO_PARAM"}]',
    ]
    result = runner.invoke(highlighter_group, cli_cmd, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "EchoStreamParamsA.frame_param.foo: FOO_PARAM" in result.stdout
