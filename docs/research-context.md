# Research Agent context and tool protocols

## Qwen3.5 native tools

For `huggingface/...Qwen3.5...` planner models, Research Agent uses the checkpoint's
native tool protocol throughout. Other backends retain the labeled Action/Action
Input protocol. Script execution and sandbox selection are unchanged.

General instructions are a system message; the research problem and data/library
context are a user message. Enabled tool definitions are passed through `tools=`
to `apply_chat_template`, which supplies native XML instructions. Definitions use
stable snake_case names (e.g. `create_script_ai`) and typed parameters with defaults
from action function signatures. Runtime arguments such as `work_dir` are not
exposed. Adding non-string arguments to the registry requires updating the type
mapping in `qwen_tools.py`.

The model writes its configured Reflection/Research Plan/Fact Check/Thought
fields as commentary, then exactly one call:

```xml
<tool_call>
<function=create_script_ai>
<parameter=datasets>
{"train": "train.csv"}
</parameter>
<parameter=objective>
Generate and validate historical features, then save features.csv.
</parameter>
<parameter=script_name>
features.py
</parameter>
<parameter=use_demand_engine>
true
</parameter>
</function>
</tool_call>
```

This is Qwen's XML-like dialect, not standards-compliant XML. String parameters
are literal text, not JSON strings or XML entities. Quotes, backslashes, internal
newlines, `<` and `&` are preserved. Object/integer parameters are decoded as JSON
and validated; booleans accept true/false and the True/False spelling emitted by
the checkpoint's history template. Duplicate/unknown parameters or functions,
missing required fields, malformed calls, mixed legacy headings, thinking/role
artifacts in commentary, multiple calls and trailing text are rejected. Reserved
protocol delimiters cannot be embedded as literal parameter data. No tool runs
on a rejected native response.

Native generation stops at `</tool_call>` using decoded generated-token suffixes,
independent of isolated stop-string tokenization. It does not stop at the legacy
`Observation:` string, which may appear inside parameters. Model EOS and output
limits still apply; incomplete calls fail validation. Thinking remains disabled.

History stores validated commentary separately from parsed arguments and raw
output. Only commentary plus one structured `tool_calls` item is replayed: the
template renders XML once, and the tool result carries its call ID. There are no
duplicated Action/Input headings or model-emitted XML in assistant content.
Rejected outputs remain in inference logs; retries send concise feedback without
replaying malformed examples. Old checkpoints reconstruct commentary from parsed
fields, exclude Action/Input fields and omit commentary containing protocol
artifacts. Invalid legacy calls become framework feedback, not tool executions.

## Summaries and context limits

- Checkpoints retain raw assistant output, canonical commentary (native mode),
  parsed arguments, call ID, raw observation, context observation and provenance.
- Observations over 5,000 characters are summarized as before, except inspected
  source remains verbatim. Summarized tool messages identify summarizer, source
  call/step and summary log. Failed/empty summaries use labeled truncation; the
  original result remains in the checkpoint.
- Research-log summaries identify source calls and distinguish plans from
  observations. Prompts preserve identifiers, metrics, paths, exceptions and
  uncertainty. These summaries remain derived evidence, not verified originals.
- Retrieval receives the latest plan. Its output is labeled memory in a user
  context message before recent exchanges, never system rules or a fabricated
  original assistant turn. Memory may overlap recent steps.
- Step/observation windows remain supported. Omitted results explicitly say they
  are omitted and are not evidence of success.
- HF budgeting counts the rendered prompt including tool definitions, reserves
  output tokens, drops oldest complete exchanges, then retrieved memory. It keeps
  system/task and the latest exchange; if these cannot fit it fails explicitly.
  Raw checkpoint evidence is not deleted.

Each HF planner log has a `.context.jsonl` sidecar containing the actual trimmed
messages, tool definitions, rendered prompt and input-token count for each attempt.
Inference logs retain decoded outputs, including rejected responses.

## Legacy action format

Other HF planners retain their existing chat-template path. Non-HF planners
receive labeled flattened text through existing APIs. HF string callers remain
supported and do not automatically enable tools.

The legacy parser decodes one leading JSON object, optionally fenced, preserves
types/escapes and excludes trailing prose from arguments. It rejects additional
immediate JSON payloads, trailing Action/Input headers, malformed JSON, duplicate
keys, nonstandard constants and mismatched keys rather than repairing them by
regex. Its required-key behavior is unchanged; native tools honor defaults from
action function signatures instead.
