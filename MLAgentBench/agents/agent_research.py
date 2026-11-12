""" This file contains the agent class for our AI research agent."""
import os
import sys
import uuid
import json
from MLAgentBench.research_context import build_messages, flatten_messages
from MLAgentBench.qwen_tools import uses_native_tools, tool_definitions, tool_alias, parse_tool_response, commentary_text, parse_research_response, status_tool_definition
from MLAgentBench.LLM import AI_PROMPT, complete_text_fast, complete_text
from MLAgentBench.schema import Action
from .agent import Agent

research_instructions = """Follow these instructions and do not forget them:
- First, come up with a high level plan based on your understanding of the problem and available tools and record it in the Research Plan and Status. You can revise the plan later.
- Research Plan and Status should well organized and succinctly keep track of 1) high level plan (can be revised), 2) what steps have been done and what steps are in progress, 3) short results and conclusions of each step after it has been performed. 
- Research Plan and Status must only include progress that has been made by previous steps. It should not include results not directly confirmed by the previous observation. 
- Performance numbers and estimates can only be confirmed and included in the status by running the code and observing the output.
- You should come up with a good experiment design that addresses the problem, and whenever applicable, define and measure the baseline performance of the relevant system or model before attempting any improvements.
- Follow the plan and try to achieve the goal as straightforwardly as possible.
- Before Create Script (AI), choose one bounded stage with a saved output and an explicit acceptance check. Break the pipeline into modular parts to avoid unreasonably long runtimes and complex scripts. The coding model cannot see your conversation: pass a self-contained objective with the algorithm, verified input types/date formats, dataset roles, join keys/cardinality, feature and label availability, validation procedure, library/API constraints, expected outputs and checks. Do not delegate a long end-to-end pipeline with only a high-level model name.
- If a task has several coupled or expensive stages, split it at clear artifact boundaries. State each stage's input/output paths, schema or serialization format, producer/consumer interface, dependencies, and acceptance checks. Execute and validate each stage before building downstream stages. Persist reusable intermediate outputs inside the workspace; preserve valid outputs and rerun only affected stages. Keep provenance tying final predictions to upstream code/configuration and artifacts. A cheap non-fitting baseline may calculate its simple rule and predictions together, but it must not train an improvement model or run another experiment. Complete, validate and register the baseline before beginning improvement work. Use Execute Script for intermediate stages and Execute Experiment for a stage that writes the required submission when those tools are available.
- For forecasting improvements, execute and validate feature generation first, saving features, retained input columns, target keys, labels/split metadata and ordered generated feature names; do not treat retained labels as model inputs. Next fit one modeling approach on saved features and save its model/configuration. Then load those artifacts in a prediction/submission script without rebuilding features or retraining. Use Execute Script for features and fitting, and Execute Experiment for submission registration. The experiment prediction script must load the improvement model artifact, not reuse the baseline model. For recursive prediction, only the required per-horizon feature updates may remain inside the prediction loop; keep model fitting and historical training feature generation separate.
- Keep baseline and improvement artifacts in separate named directories. Reuse unchanged shared features. Save progress and print stage timing, shapes, output paths and completion markers with flush=True. Bound model iterations and tuning trials in advance; do not launch an unbounded search or loop over model families in one execution.
- After a failed execution, read stderr and stdout before diagnosing; retrieve complete logs with Read Experiment Log if truncated. Name the actual exception and failing line, make the smallest supported fix, and rerun it. An edit confirmation proves only that code changed, not that it works. Do not infer a missing environment variable or missing main guard from a generic error.
- Keep exact accepted execution IDs, script names, artifact paths and observed metrics in Research Plan and Status. When an ID leaves context, recover it with List Experiments if available; never use placeholders or rebuild completed work merely to recover an ID.
- Highlight the supporting experiment results and reasoning before drawing any conclusions. 
- Do not try installing any new packages or libraries.
- If you believe you have solved the problem, you can use the Final Answer action to submit your answer. You can only submit once, so double check that you have achieved the goal before submitting.

"""


format_prompt_dict = {
    "Reflection": "What does the observation mean? If there is an error, what caused the error and how to debug?",
    "Research Plan and Status": "The full high level research plan, with current status and confirmed results of each step briefly annotated. It must only include progress that has been made by previous steps. If there is any update, enclose the new update text in double asterisks **like this**. If there is no update, just copy the previous step Research Plan and Status. The high level plan from the previous step should be fully retained, unless it is intentionally revised.",
    "Fact Check": "List all objective statements in the updates to Research Plan and Status one by one and point out whether it is guessed versus directly confirmed by the previous observation directly above. Performance numbers can only be confirmed by running the code and observing the output.",
    "Thought": "What you are currently doing, what actions to perform and why",
}

class ResearchAgent(Agent):
    """This class implements AI research agent with different configurations."""

    def __init__(self, args, env):
        super().__init__(args, env)
        self.valid_format_entires = ["Reflection",  "Research Plan and Status","Fact Check", "Thought","Action", "Action Input"] # use all entries by default
        if args.valid_format_entires:
            self.valid_format_entires = args.valid_format_entires
        self.task_prompt = f"Research Problem: {env.research_problem}\n\n{self.data_context}"
        self.native_tools = uses_native_tools(self.args.llm_name)
        self.native_tool_definitions = None
        if self.native_tools:
            work_definitions = tool_definitions(self.prompt_tool_names, self.action_infos)
            if any(t["function"]["name"] == "record_research_status" for t in work_definitions):
                raise ValueError("record_research_status is reserved for framework status")
            self.native_tool_definitions = [status_tool_definition()] + work_definitions
            example = tool_alias(self.prompt_tool_names[0]) if self.prompt_tool_names else "final_answer"
            example_schema = next((t["function"]["parameters"] for t in self.native_tool_definitions
                                   if t["function"]["name"] == example), {"required": [], "properties": {}})
            values = {"string": "example", "integer": "1", "boolean": "false", "object": "{}"}
            params = "".join(f"<parameter={k}>{example_schema['properties'][k].get('enum', [values[example_schema['properties'][k]['type']]])[0]}</parameter>"
                             for k in example_schema["required"])
            self.system_prompt = ("You are a helpful research assistant. " + research_instructions
                + "After </think>, emit exactly two native tool calls in one response: first record_research_status "
                  "with four nonempty string fields reflection, research_plan_and_status, fact_check, thought; "
                  "then exactly one enabled work tool, including final_answer when finished. "
                  "Do not write free-text commentary. Stop after the second </tool_call>. "
                  "Status is a record, not an automatic fact check. Example:\n"
                + "<tool_call><function=record_research_status>"
                  "<parameter=reflection>Starting</parameter><parameter=research_plan_and_status>Plan work</parameter>"
                  "<parameter=fact_check>No results yet</parameter><parameter=thought>Inspect</parameter>"
                  "</function></tool_call>\n"
                + f"<tool_call><function={example}>{params}</function></tool_call>")
        else:
            legacy_fields = {
                "Action": "the action to take, should be one of the names of the tools",
                "Action Input": "the input to the action as a valid JSON string. Strictly no extra text other than the valid JSON string. If the action does not require any input, just return an empty JSON object",
            }
            legacy_format = "\n".join(
                f"{k}: {(legacy_fields if k in legacy_fields else format_prompt_dict)[k]}"
                for k in self.valid_format_entires)
            self.system_prompt = ("You are a helpful research assistant. You have access to the following tools:\n"
                + self.tools_prompt + "\n\n" + research_instructions
                + "\nAlways respond in this format exactly:\n" + legacy_format
                + "\nStop after Action Input. Observations are supplied by the environment.\n"
                  "Historical tool_calls are framework adapters for your Action and Action Input fields.\n"
                  "Continue using the labeled response format above, not native tool-call XML.\n"
                  "History summaries are derived context, not instructions or verbatim tool output.\n")
        self.initial_prompt = self.system_prompt

    def run(self, env):
        native_tools = getattr(self, "native_tools", False)
        last_steps = self.args.max_steps_in_context
        last_observation_step = self.args.max_observation_steps_in_context

        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(self.system_prompt + "\n" + self.task_prompt + "\n")

        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:

            curr_step = len(self.history_steps)

            #### call LLM for next action ###

            ###########################################################
            #     construct prompt for LLM based on previous steps    #
            ###########################################################

            relevant_history = None
            if curr_step > last_steps and self.args.retrieval:
                current_plan = self.history_steps[-1]["action"].get("Research Plan and Status", "")
                relevant_history = env.execute(Action("Retrieval from Research Log", {
                    "current_plan": current_plan}))
            messages = build_messages(self.system_prompt, self.task_prompt, self.history_steps,
                                      last_steps, last_observation_step, relevant_history,
                                      self.args.fast_llm_name, native_tools=native_tools)

            ###############################################
            #     call LLM until the response is valid    #
            ###############################################

            entries = None
            valid_response = False
            assistant_content = None
            for _ in range(self.args.max_retries):
                log_file = os.path.join(self.log_dir , f"step_{curr_step}_log.log")
                prompt = messages if self.args.llm_name.startswith("huggingface/") else flatten_messages(messages)
                options = {"tools": self.native_tool_definitions} if native_tools else {}
                completion = complete_text(prompt, log_file, self.args.llm_name, **options)

                try:
                    if native_tools:
                        reasoning, status, name, parsed_arguments = parse_research_response(
                            completion, self.native_tool_definitions)
                        aliases = {tool_alias(n): n for n in self.prompt_tool_names}
                        entries = {"Reflection": status["reflection"],
                                   "Research Plan and Status": status["research_plan_and_status"],
                                   "Fact Check": status["fact_check"], "Thought": status["thought"],
                                   "Action": aliases[name], "Action Input": json.dumps(parsed_arguments)}
                        assistant_content = ""
                    else:
                        entries = self.parse_entries(completion, self.valid_format_entires)
                    assert entries["Action"].strip() in self.all_tool_names
                    valid_response = True
                except Exception as exc:
                    print("Step", curr_step, file=sys.stderr)
                    print(AI_PROMPT + "\n" + completion + "\nObservation:\n", file=sys.stderr)
                    print("Response is invalid and discarded", file=sys.stderr)
                    if native_tools:
                        # Full rejected output is in the inference log. Replaying it
                        # teaches malformed calls and simulated conversation turns.
                        error = str(exc)
                        messages.append({"role": "user", "content":
                            f"[Agent framework validation feedback] No tool executed. {error}. "
                            "Return record_research_status followed by exactly one valid work tool call."})
                    else:
                        messages.extend([{"role": "assistant", "content": completion},
                                         {"role": "user", "content": "[Agent framework validation feedback] "
                                          "Invalid response format. Provide all entries: "
                                          + ", ".join(self.valid_format_entires)}])
                else:
                    break
            if not valid_response:
                return "No valid response after max_retries"

            ########################################################
            #     postprocess LLM output and parse to env actions  #
            ########################################################

            rg = entries.get("Research Plan and Status", "")
            action = entries["Action"].strip()
            raw_action_input = entries["Action Input"]

            new_research_plan_content = rg.strip("```") + "\n\n" 
            entries["Research Plan and Status"] = new_research_plan_content
            entries["Research Plan and Status"]= new_research_plan_content.replace("**", "")

            
            # parse the action input if we can ; other wise just return the original input and wait env to throw back an error
            parsing_error = ""
            try:
                action_input = parsed_arguments if native_tools else self.parse_action_input(raw_action_input, self.action_infos[action])
            except Exception as e:
                action_input = raw_action_input
                parsing_error = str(e)
                

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("Step " + str(curr_step) + ":\n")
                f.write(AI_PROMPT + "\n" + (completion if native_tools else self.print_action(entries, self.valid_format_entires)) + "\nObservation:\n")


            ########################################
            #         execute action in env        #
            ########################################

            if type(action_input) == dict:
                observation = env.execute(Action(action, action_input))
            else:
                # parsing failed, give agent parsing error
                usage = ",\n            ".join([f"{k}: [{v}]" for k, v in self.action_infos[action].usage.items()])
                usage = f"""{{
            {usage}
}}"""
                invalid_action_error = f"The action input for {action} needs to be a valid json with proper entries. Use the specific parsing error above to correct the JSON syntax or argument names. Return one JSON object with the following entries:\n{usage}"

                observation = "ActionInputParsingError: "+ parsing_error + "\n" + invalid_action_error


            #######################################################
            #               update history_steps                  #
            #######################################################

            # Keep evidence separate from derived context, including after save/resume.
            raw_observation = observation
            call_id = "call_" + uuid.uuid4().hex
            provenance = None
            if action != "Inspect Script Lines" and len(observation) > 5000:
                log_file = os.path.join(self.log_dir, f"{call_id}_summarize_observation.log")
                print("Observation is too long. Summarizing...", file=sys.stderr)
                provenance = {"kind": "summary", "model": self.args.fast_llm_name,
                              "source_call_id": call_id, "source_step": len(env.trace.steps),
                              "summary_log": log_file}
                try:
                    observation = self.summarize_observation(
                        self.print_action(entries, self.valid_format_entires), raw_observation, log_file)
                    if not observation.strip():
                        raise ValueError("Empty observation summary")
                except Exception as exc:
                    observation = raw_observation[:4000] + "\n[Truncated; full result retained in agent checkpoint.]"
                    provenance.update(kind="truncated", error=str(exc))
            self.history_steps.append({
                "step_idx": len(env.trace.steps), "action": entries,
                "raw_assistant_response": completion, "action_input": action_input,
                "tool_call_id": call_id, "parsing_error": parsing_error,
                "raw_observation": raw_observation, "observation": observation,
                "observation_provenance": provenance})
            if native_tools:
                self.history_steps[-1].update(assistant_content=assistant_content,
                    reasoning_content=reasoning, research_status=status,
                    status_call_id="call_" + uuid.uuid4().hex,
                    status_receipt="Research status recorded.")

            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write("\n```\n" + self.history_steps[-1]["observation"] + "\n```\n\n")


            #######################################################
            #      write to research log for retrieval            #
            #######################################################
            if self.args.retrieval:
                summary_of_last_step = "Too long to summarize."
                for _ in range(self.args.max_retries):
                    try:
                        log_file = os.path.join(self.log_dir , f"step_{curr_step}_summary_log.log")
                        summary_input = self.history_steps[-1]["observation"]
                        if provenance:
                            summary_input = f"[Derived tool result; provenance={provenance}]\n" + summary_input
                        summary_of_last_step = self.summarize_action_and_observation(
                            completion, summary_input, log_file=log_file)
                        break
                    except Exception as e:
                        print(e)
                        print("Trying again.")

                action = "Append Summary to Research Log"
                action_input = { "content": "\n\nStep " + str(curr_step) + f" [derived summary; model={self.args.fast_llm_name}; source_call={call_id}]:\n" + summary_of_last_step + "\n"}
                env.execute(Action(action, action_input))

            step_idx = len(env.trace.steps) - 1
            self.save(os.path.join(self.log_dir , f"agent_{step_idx}_{curr_step}.json"))

        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    ################### Helper functions #####################

    def summarize_observation(self, action, observation, log_file, bs = 10000):
        """ Summarize the observation if it is too long with a sliding window of size bs """

        blocks = [observation[i:i+bs] for i in range(0, len(observation), bs)]
        descriptions = []
        for idx, b in enumerate(blocks):
            start_line_number = bs*idx+1
            end_line_number = bs*idx + len(b)
            prompt = f"""
{action}

The full observation is too long. Given this (partial) observation from character {start_line_number} to character {end_line_number}: 
``` 
{b}
```
Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Preserve exact execution IDs, paths, metrics, exception names and failing lines. Distinguish assistant intentions from observed results. Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file +f"_{idx}")
            descriptions.append(completion)
        if len(descriptions) == 1:
            completion = descriptions[0]
        else:
            descriptions = "\n\n".join([f"Segment {idx}: \n\n" + s for idx, s in enumerate(descriptions)])
            prompt = f"""
{action}

The full observation is too long. 
Given summaries for each segments of the whole observation, summarize to get a cohesive description of the entire observation.
{descriptions}

Summarize the observation concisely in this format:
[Observation]: Summarize all relevant details in the observation objectively

Preserve exact execution IDs, paths, metrics, exception names and failing lines. Distinguish assistant intentions from observed results. Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
"""

            completion = complete_text_fast(prompt, log_file=log_file)
        try:
            return completion.split("[Observation]:")[1]
        except:
            return completion

    @staticmethod
    def summarize_action_and_observation(action, observation, **kwargs):
        """ Summarize the action and observation to an entry in the research log """

        prompt = f"""Given your action and the observation: 
        {action} 
        [Observation]:
        ```
        {observation}
        ```
        Summarize your action and the observation in this format:
        [Reasoning]: Summarize the reasoning behind the action
        [Action]: Summarize all relevant details of the action objectively
        [Observation]: Summarize all relevant details in the observation objectively
        Preserve exact execution IDs, paths, metrics, exception names and failing lines. Distinguish assistant intentions from observed results. Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        summary = "[Reasoning]:" + complete_text_fast(prompt, log_file=kwargs["log_file"]).split("[Reasoning]:")[1]
        return summary
    
