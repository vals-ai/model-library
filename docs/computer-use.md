# Computer use

OpenAI, Anthropic and Google native computer tools are supported. Other providers reject them.

## Results

`result.computer_calls` lists the actions the model wants. Each call has its provider,
call ID, actions, and pending safety checks. An OpenAI call can hold several actions;
an Anthropic or Google call holds one. Run the actions as the provider defines them.

## A session

Run the actions, take a screenshot, and send it back with the call ID and the
previous `result.history`. Each turn replays the history, so the provider stores nothing.

Your code decides each pending safety check. Send back only the checks that it approves.
The gateway does not approve a check for you.

```python
from model_library import model
from model_library.base import (
    ComputerCallOutput,
    TextInput,
    ToolDefinition,
)

llm = model("openai/gpt-6.1-sol")
tool = ToolDefinition(name="computer", body={"type": "computer"})

result = await llm.query([TextInput(text="Open the document")], tools=[tool])
call = result.computer_calls[0]
approved_checks = [c for c in call.pending_safety_checks if operator_approves(c)]

result = await llm.query(
    [ComputerCallOutput(
        call_id=call.call_id,
        screenshot_url="data:image/png;base64,...",
        acknowledged_safety_checks=approved_checks,
    )],
    tools=[tool],
    history=result.history,
)
```

## Provider notes

- OpenAI: the tool body is `{"type": "computer"}`. Declare the tool from the first turn.
  Chat Completions is not supported.
- Anthropic: the model config sets the tool version (`computer_tool_type`). Claude 5.5 models
  need the newer computer toolset, which is not supported yet.
  The tool body also needs `display_width_px` and `display_height_px`.
  Other tool options, such as `enable_zoom`, are sent only when you set them.
  A reply can have text, a screenshot, or both.
- Google: the tool body is `{"type": "computer"}`. It sets `environment` (`desktop`, the
  default, `browser` or `mobile`) and `excluded_predefined_functions`. Other keys are not sent.
  Each action is `{"name": ..., "args": ...}`, one predefined UI function call; any call to a
  name that is not a declared function tool is an action. `x` and `y` are on a 0-999 grid.
  A `safety_decision` in the args is the pending safety check. The proxy only forwards the
  caller's acknowledgement; the executor must get approval for `require_confirmation`.
  Signed history does not establish operator approval. Screenshots must be `data:` URLs,
  not HTTPS URLs.
- Do not change the history between turns. The gateway signs it.

## Through the gateway

- Send screenshots as `image_url`: a `data:` URL, or an HTTPS URL except for Google.
- Usage records keep no actions, screenshots, or safety-check text.
