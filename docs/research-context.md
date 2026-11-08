# Research Agent context attribution

The Hugging Face Research Agent passes a message list to the model's chat
template. General instructions and tool descriptions are `system` content;
the research problem and data/library context are `user` content. Each retained
step has an `assistant` message and a corresponding `tool` response.

The generation contract remains the existing labeled Reflection/Plan/Fact
Check/Thought/Action/Action Input format. The existing parser and action registry
still validate and execute actions. History adapts validated actions into
`tool_calls` with parsed argument dictionaries and stable call IDs; these are
framework metadata, not an assertion that the model originally emitted native
tool-call XML. The original assistant response is retained verbatim in content,
including its Action/Input fields. No native tools schema is passed that would
instruct the model to switch output protocols. Qwen's template renders the
historical call and result tags; native thinking remains disabled.

Invalid arguments do not create a fake tool execution: they retain the assistant
response followed by explicitly labeled framework feedback in a user message.
Format retries likewise retain the rejected response and validation feedback.
Argument failures remain in history rather than being silently deleted after a
later success; they count toward the step limit.

## Summaries and retention

- Checkpoints retain the raw assistant response, parsed action input, call ID,
  raw observation, context observation, and summary provenance separately.
- Observations over 5,000 characters are summarized as before, except inspected
  source remains verbatim. Summarized tool messages identify the summarizer,
  source call/step and summary log. A failed/empty summary falls back to labeled
  truncation; the raw result remains in the checkpoint.
- Research-log entries identify their summarizer and source call. Summarization
  prompts distinguish plans from observations and preserve identifiers, paths,
  metrics, exceptions and uncertainty. They remain model-generated summaries,
  not independently verified evidence.
- Retrieval receives the latest plan. Its result is explicitly labeled derived
  memory in a user context message before recent exchanges, never system rules
  or a fabricated original assistant turn. Memory may overlap recent steps.
- Existing step and observation windows remain supported. Omitted observations
  explicitly say they are omitted and are not evidence of success.
- HF token budgeting uses the tokenizer/model context limit, reserves output
  tokens, drops oldest complete exchanges, then drops retrieved memory. It keeps
  the system/task and latest exchange. If those cannot fit, it fails explicitly
  instead of slicing through messages or silently truncating evidence.

Old checkpoints remain readable. Their assistant text is reconstructed from
stored entries; old observations are labeled as possibly already summarized.
Raw data lost by an older checkpoint cannot be reconstructed. Parsed legacy JSON
arguments get matched call/result pairs; other legacy entries use labeled
framework feedback rather than inventing arguments.

## Inspection and compatibility

Each HF planner log has a `.context.jsonl` sidecar recording the actual messages
after token-budget trimming, rendered template text, and input-token count for
each attempt. This is the authoritative input log; the traditional log remains
available for human-readable output and legacy token estimates.

String callers of the HF helper remain supported. Non-HF Research Agent
backends receive explicitly labeled flattened text through their existing APIs;
this change does not implement native message/tool adapters for those backends.
HF models used for Research Agent must provide a chat template supporting the
roles/tool metadata; unsupported templates fail rather than silently flattening.

Script execution, execution-mode selection, Bubblewrap isolation, unsandboxed
execution and provenance enforcement are unchanged.
