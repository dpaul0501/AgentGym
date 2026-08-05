"""TravelSageAgent: a real LangGraph tool-calling agent (real Bedrock Claude, real mock tools)
onboarded onto AgentGym's Agent protocol. consumes() declares the two scopes this demo binds —
MEMORY and GUARDRAILS — with no default (None), so an unbound scope means "raw, unmodified tool
behavior": no compression, no guardrail check. That's the deliberate "before" state both demo
scripts start from; binding a real Artifact for either scope changes real agent behavior on the
next run, not just a reported score.
"""

from __future__ import annotations

import json
import uuid

from headroom.compress import CompressConfig, compress
from invariant.analyzer import Policy
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool

from agentgym.core.artifact import Artifact
from agentgym.core.protocols import Task
from agentgym.core.scope import GUARDRAILS, MEMORY
from agentgym.core.trace import Span, Trace
from examples.travel_sage_agent.llm import build_bedrock_llm
from examples.travel_sage_agent.tools import ALL_TOOLS

BLOCKED_MESSAGE = {"blocked": True, "reason": "guardrail policy violation — tool output withheld"}


def _apply_memory(raw_output, memory_artifact: Artifact | None):
    if memory_artifact is None:
        return raw_output
    config = CompressConfig(
        compress_user_messages=memory_artifact.value.get("compress_user_messages", False),
        protect_recent=memory_artifact.value.get("protect_recent", 4),
        min_tokens_to_compress=memory_artifact.value.get("min_tokens_to_compress", 250),
    )
    model = memory_artifact.value.get("model", "claude-sonnet-4-5-20250929")
    messages = [{"role": "tool", "content": json.dumps(raw_output)}]
    result = compress(messages, model=model, config=config)
    return result.messages[0]["content"]


def _apply_guardrail(tool_name: str, tool_args: dict, raw_output, guardrails_artifact: Artifact | None):
    if guardrails_artifact is None:
        return raw_output
    policy = Policy.from_string(guardrails_artifact.value["policy_dsl"])
    call_id = uuid.uuid4().hex[:8]
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(raw_output)},
    ]
    result = policy.analyze(messages)
    if result.errors:
        return {**BLOCKED_MESSAGE, "detail": str(result.errors[0])}
    return raw_output


def _wrap_tool(base_tool: BaseTool, memory_artifact: Artifact | None, guardrails_artifact: Artifact | None) -> BaseTool:
    @tool(base_tool.name, args_schema=base_tool.args_schema, description=base_tool.description)
    def wrapped(**kwargs):
        raw_output = base_tool.func(**kwargs)
        checked = _apply_guardrail(base_tool.name, kwargs, raw_output, guardrails_artifact)
        if checked is not raw_output:  # a guardrail violation short-circuits memory compression
            return checked
        return _apply_memory(raw_output, memory_artifact)

    return wrapped


def _messages_to_spans(messages, trace_id: str) -> list[Span]:
    spans: list[Span] = []
    pending_calls: dict[str, dict] = {}
    for i, msg in enumerate(messages):
        if msg.type == "ai":
            for call in getattr(msg, "tool_calls", None) or []:
                pending_calls[call["id"]] = call
            spans.append(Span(
                span_id=f"s{i}", parent_span_id=None, trace_id=trace_id, name="llm_turn",
                kind="llm", input={}, output={"content": msg.content}, start_time=float(i),
                end_time=float(i) + 0.1, attributes={},
            ))
        elif msg.type == "tool":
            call = pending_calls.get(msg.tool_call_id, {})
            spans.append(Span(
                span_id=f"s{i}", parent_span_id=None, trace_id=trace_id,
                name=call.get("name", "unknown_tool"), kind="tool",
                input=call.get("args", {}), output=msg.content, start_time=float(i),
                end_time=float(i) + 0.1, attributes={},
            ))
    return spans


class TravelSageAgent:
    def __init__(self, llm=None, model_env_var: str = "BEDROCK_MODEL_SONNET"):
        """llm: any real LangChain chat model (ChatBedrockConverse, ChatOllama, ...). Defaults to
        Bedrock Claude via model_env_var when not given, so existing callers are unaffected."""
        self.llm = llm if llm is not None else build_bedrock_llm(model_env_var)

    def consumes(self) -> dict:
        return {MEMORY: None, GUARDRAILS: None}

    def run(self, task: Task, artifacts: dict) -> Trace:
        wrapped_tools = [
            _wrap_tool(t, artifacts.get(MEMORY), artifacts.get(GUARDRAILS)) for t in ALL_TOOLS
        ]
        app = create_agent(self.llm, wrapped_tools)
        result = app.invoke({"messages": [("user", task.instruction)]})

        trace_id = f"t-{task.task_id}"
        spans = _messages_to_spans(result["messages"], trace_id)
        final_content = result["messages"][-1].content
        return Trace(
            trace_id=trace_id, task_id=task.task_id, spans=spans,
            metadata={"final_response": final_content},
        )
