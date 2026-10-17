"""Real vLLM/tokenizer regressions; run in isolated CPU process using the installed parser."""
import json
import unittest
from base_regression import Regressions as ExistingRegressions, parse_stream, call, TOKENIZER, REQUEST
from vllm.parser.qwen3 import Qwen3Parser
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

class Names(unittest.TestCase):
    def variants(self, text):
        for token_mode in (False, True):
            for size in (1,2,7,31,100000):
                yield parse_stream(text,size,False,token_mode)

    def test_known_namespace_and_whitespace(self):
        for name in ('exec_command',' exec_command ','functions.exec_command','tools.exec_command',' functions.exec_command '):
            text='Now the rewrite:\n'+call('echo fixture').replace('<function=exec_command>', '<function='+name+'>')
            for _,_,calls in self.variants(text):
                self.assertEqual(len(calls),1)
                self.assertEqual(calls[0]['name'],'exec_command')
                self.assertEqual(json.loads(calls[0]['arguments']),{'cmd':'echo fixture'})
            parser=Qwen3Parser(TOKENIZER,tools=REQUEST.tools,chat_template_kwargs={'enable_thinking':False})
            result=parser.extract_tool_calls(text,REQUEST)
            self.assertEqual(result.tool_calls[0].function.name,'exec_command')
            self.assertEqual(json.loads(result.tool_calls[0].function.arguments),{'cmd':'echo fixture'})

    def test_unknown_namespace_not_matched(self):
        for name in ('other.exec_command','tools.functions.exec_command','functions.unknown','exec_command_extra'):
            text=call('echo fixture').replace('<function=exec_command>', '<function='+name+'>')
            for _,_,calls in self.variants(text):
                self.assertEqual(calls,[])

    def test_exact_qualified_name_wins(self):
        tools=[{'type':'function','name':n,'parameters':{'type':'object','properties':{'cmd':{'type':'string'}}}} for n in ('exec_command','functions.exec_command')]
        req=ResponsesRequest(model='qwen',input='test',tools=tools,tool_choice='auto')
        parser=Qwen3Parser(TOKENIZER,tools=req.tools,chat_template_kwargs={'enable_thinking':False})
        text=call('echo fixture').replace('<function=exec_command>','<function=functions.exec_command>')
        result=parser.extract_tool_calls(text,req)
        self.assertEqual(result.tool_calls[0].function.name,'functions.exec_command')

    def test_schema_coercion_uses_resolved_name(self):
        text=call('echo fixture').replace('<function=exec_command>','<function=functions.exec_command>').replace('\n</function>','\n<parameter=yield_time_ms>1000</parameter>\n</function>')
        for _,_,calls in self.variants(text):
            self.assertEqual(json.loads(calls[0]['arguments'])['yield_time_ms'],1000)

if __name__=='__main__':
    unittest.main(verbosity=2)
