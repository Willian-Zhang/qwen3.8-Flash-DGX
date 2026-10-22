"""v0.32 wire contracts with real producers and synthetic engine frames."""
import asyncio
import json
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'issue31_safe_output/tests'), str(root / 'issue31_schema_arguments')]
import producer_tests as producer


class ResponsesV032(unittest.TestCase):
    def test_native_reasoning_event_names_and_item_ids(self):
        events, _ = asyncio.run(producer.actual_stream('<think>Fixture.</think>Answer', 'stop', 1))
        types = [event.type for event in events]
        self.assertIn('response.reasoning_text.delta', types)
        self.assertIn('response.reasoning_text.done', types)
        self.assertNotIn('response.reasoning_summary_text.delta', types)
        self.assertEqual(events[-1].type, 'response.completed')
        added = [event.item.id for event in events if event.type == 'response.output_item.added']
        done = [event.item.id for event in events if event.type == 'response.output_item.done']
        final = [item.id for item in events[-1].response.output]
        self.assertEqual(added, done)
        self.assertEqual(done, final)

    def test_tool_item_ids_match_terminal_response(self):
        text = 'Fixture.</think><tool_call><function=exec_command><parameter=cmd>fixture</parameter></function></tool_call>'
        for size in (1, 7, 100000):
            events, _ = asyncio.run(producer.actual_stream(text, 'stop', size))
            self.assertEqual(events[-1].type, 'response.completed')
            added = [e.item for e in events if e.type == 'response.output_item.added' and e.item.type == 'function_call']
            final = [i for i in events[-1].response.output if i.type == 'function_call']
            self.assertEqual(len(added), 1)
            self.assertEqual([(i.id, i.call_id) for i in added], [(i.id, i.call_id) for i in final])
            self.assertEqual(json.loads(final[0].arguments), {'cmd': 'fixture'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
