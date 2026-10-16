"""CPU regressions with the deployed model tokenizer; no model execution."""
import json
import os
import unittest

from transformers import AutoTokenizer
from vllm.parser.qwen3 import Qwen3Parser
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

MODEL = os.environ['QWEN_TOKENIZER']
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
TOOLS = [{"type": "function", "name": "exec_command", "description": "Execute a command.",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"},
          "yield_time_ms": {"type": "integer"}}, "required": ["cmd"]}}]
REQUEST = ResponsesRequest(model="qwen", input="test", tools=TOOLS, tool_choice="auto")


def call(cmd, closed_param=True):
    return ("<tool_call>\n<function=exec_command>\n<parameter=cmd>" + cmd
            + ("</parameter>" if closed_param else "") + "\n</function>\n</tool_call>")


def parse_stream(text, size, thinking=False, token_mode=False, prompt_open=True):
    parser = Qwen3Parser(TOKENIZER, tools=REQUEST.tools,
                         chat_template_kwargs={"enable_thinking": thinking})
    prompt = "<|im_start|>assistant\n" + ("<think>\n" if thinking and prompt_open else "")
    prompt_ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    if token_mode:
        ids = TOKENIZER.encode(text, add_special_tokens=False)
        chunks = []
        previous = ""
        for i in range(0, len(ids), size):
            current = TOKENIZER.decode(ids[:i+size], skip_special_tokens=False)
            assert current.startswith(previous)
            chunks.append((current[len(previous):], ids[i:i+size]))
            previous = current
    else:
        chunks = [(text[i:i+size], []) for i in range(0, len(text), size)]
    content, reasoning, calls = [], [], {}
    for i, (chunk, ids) in enumerate(chunks):
        delta = parser.parse_delta(delta_text=chunk, delta_token_ids=ids,
                                   request=REQUEST, prompt_token_ids=prompt_ids if i == 0 else None,
                                   finished=i == len(chunks)-1)
        if delta is None:
            continue
        content.append(delta.content or "")
        reasoning.append(delta.reasoning or "")
        for tc in delta.tool_calls or []:
            entry = calls.setdefault(tc.index, {"name": "", "arguments": ""})
            if tc.function:
                entry["name"] += tc.function.name or ""
                entry["arguments"] += tc.function.arguments or ""
    return "".join(content), "".join(reasoning), list(calls.values())


class Regressions(unittest.TestCase):
    def variations(self, text, thinking=False):
        for token_mode in (False, True):
            for size in (1, 2, 7, 31, 100000):
                yield parse_stream(text, size, thinking, token_mode)

    def test_protocol_tokens_never_leak(self):
        for content, reasoning, calls in self.variations("</think>OK.<|im_end|>", True):
            self.assertEqual(content.strip(), "OK.")
            self.assertEqual(calls, [])

    def test_normal_text_survives_special_tokens(self):
        for content, _, calls in self.variations("before<|im_end|>after"):
            self.assertEqual(content, "beforeafter")
            self.assertEqual(calls, [])

    def test_closed_call_with_unclosed_last_parameter(self):
        cmd = "python3 -c 'print(\"a\\\\b\")'\n# second line"
        for _, _, calls in self.variations(call(cmd, False)):
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": cmd})

    def test_full_escaped_command_and_typed_parameter(self):
        cmd = "python3 - <<'PY'\nprint('quotes: \\\" \\\\')\nPY"
        text = call(cmd).replace("\n</function>", "\n<parameter=yield_time_ms>1000</parameter>\n</function>")
        for _, _, calls in self.variations(text):
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": cmd, "yield_time_ms": 1000})

    def test_truncated_command_is_not_completed(self):
        text = "<tool_call>\n<function=exec_command>\n<parameter=cmd>echo danger"
        for _, _, calls in self.variations(text):
            for tc in calls:
                self.assertNotIn("cmd", json.loads(tc["arguments"] or "{}"))

    def test_quoted_reasoning_is_preserved_without_phantom_call(self):
        example = "Example only:\n" + call("echo quoted") + "\nBack to reasoning."
        text = example + "</think>Done."
        for content, reasoning, calls in self.variations(text, True):
            self.assertEqual(content, "Done.")
            self.assertEqual(reasoning, example)
            self.assertEqual(calls, [])

    def test_real_call_after_quoted_reasoning(self):
        text = "Example:\n" + call("echo quoted") + "\n</think>" + call("echo real")
        for _, _, calls in self.variations(text, True):
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": "echo real"})

    def test_implicit_reasoning_end_still_calls_tool(self):
        for _, _, calls in self.variations("I will use a tool.\n" + call("echo real"), True):
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": "echo real"})

    def test_fenced_example_is_text(self):
        text = "```xml\n" + call("echo quoted") + "\n```\nDone."
        for content, _, calls in self.variations(text):
            self.assertEqual(content, text)
            self.assertEqual(calls, [])

    def test_unpromoted_invalid_tool_markup_is_preserved_in_order(self):
        text = "A <tool_call>garbage</tool_call> B"
        for content, _, calls in self.variations(text):
            self.assertEqual(content, text)
            self.assertEqual(calls, [])

    def test_real_call_after_malformed_opener(self):
        text = "Use <tool_call>\n<function=not_a_real_tool\nThen:\n" + call("echo real")
        for _, _, calls in self.variations(text):
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": "echo real"})

    def test_tool_markup_inside_command_is_preserved(self):
        cmd = "echo '<tool_call> <|im_end|> quoted'"
        # Turn control tokens are suppressed in output; ordinary XML markers
        # within parameters must remain verbatim.
        cmd = cmd.replace("<|im_end|> ", "")
        for _, _, calls in self.variations(call(cmd)):
            self.assertEqual(json.loads(calls[0]["arguments"]), {"cmd": cmd})


if __name__ == "__main__":
    unittest.main(verbosity=2)
