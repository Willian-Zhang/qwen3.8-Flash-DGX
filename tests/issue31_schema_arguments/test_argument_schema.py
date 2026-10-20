"""Synthetic actual ParserManager and schema-authoritative argument regressions."""
import json,unittest
import test_parser_safety as f
from vllm.parser.engine.argument_validation import InvalidToolArguments,validate_or_unwrap,find_parameters
SCHEMA={'type':'object','properties':{'cmd':{'type':'string'},'count':{'type':'integer'}},'required':['cmd'],'additionalProperties':False}
TOOLS=[{'type':'function','name':'exec_command','parameters':SCHEMA}]
def xml(value,name='exec_command',parameter='arguments'):
 return '<tool_call>\n<function='+name+'>\n<parameter='+parameter+'>\n'+value+'\n</parameter>\n</function>\n</tool_call>'
class ArgumentSchema(unittest.TestCase):
 def paths(self,text,tools=TOOLS,thinking=False):
  for size in (1,7,100000):
   result,p=f.streaming(text,size,thinking=thinking,tools=tools)
   yield [json.loads(c['arguments']) for c in result[2]],p
  result,p=f.full(text,thinking=thinking,tools=tools)
  yield [json.loads(c.arguments) for c in result[2] or []],p
 def test_real_shape_json_string_wrapper_normalized_all_paths(self):
  value={'cmd':'printf fixture'}
  for args,p in self.paths(xml(json.dumps(value))):self.assertEqual(args,[value]);self.assertFalse(p.has_rejected_tool_calls)
 def test_escaped_newline_backslash_and_unicode_preserved(self):
  value={'cmd':'printf \'C:\\tmp\\file "日本語"\'\nnext-line\n','count':3}
  for args,p in self.paths(xml(json.dumps(value,ensure_ascii=False))):self.assertEqual(args,[value])
 def test_one_layer_object_wrapper(self):
  original={'arguments':{'cmd':'fixture'}}
  self.assertEqual(json.loads(validate_or_unwrap(json.dumps(original),SCHEMA)),{'cmd':'fixture'})
 def test_double_encoded_inner_not_recursively_unrolled(self):
  for args,p in self.paths(xml(json.dumps(json.dumps({'cmd':'fixture'})))):
   self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_non_json_and_scalar_wrappers_rejected(self):
  for text in ['not JSON','[]','42','null','{"cmd":"fixture"} trailing']:
   for args,p in self.paths(xml(text)):
    self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_missing_required_inner_rejected(self):
  for args,p in self.paths(xml('{}')):self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_extra_properties_inner_rejected(self):
  for args,p in self.paths(xml('{"cmd":"fixture","unexpected":1}')):self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_inner_type_mismatch_not_coerced(self):
  for args,p in self.paths(xml('{"cmd":17}')):self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_legitimate_arguments_property_is_not_unwrapped(self):
  schema={'type':'object','properties':{'arguments':{'type':'string'}},'required':['arguments'],'additionalProperties':False}
  tools=[{'type':'function','name':'exec_command','parameters':schema}]
  value='{"cmd":"this is a string payload"}'
  for args,p in self.paths(xml(value),tools):self.assertEqual(args,[{'arguments':value}]);self.assertFalse(p.has_rejected_tool_calls)
 def test_valid_original_keeps_exact_serialization(self):
  original=' { "cmd" : "fixture" } '
  self.assertEqual(validate_or_unwrap(original,SCHEMA),original)
 def test_arguments_with_sibling_never_unwrapped(self):
  with self.assertRaises(InvalidToolArguments):validate_or_unwrap('{"arguments":{"cmd":"fixture"},"other":1}',SCHEMA)
 def test_original_valid_arguments_even_when_inner_would_match(self):
  schema={'type':'object','properties':{'arguments':{},'cmd':{'type':'string'}},'additionalProperties':False}
  original='{"arguments":{"cmd":"fixture"}}';self.assertEqual(validate_or_unwrap(original,schema),original)
 def test_namespace_uses_exact_offered_schema(self):
  tools=[{'type':'namespace','name':'functions_group','description':'Fixture namespace','tools':[{'type':'function','name':'run','description':'fixture','parameters':SCHEMA}]}]
  for name in ['functions_group.run','functions_group__run','functions.functions_group.run']:
   for args,p in self.paths(xml('{"cmd":"fixture"}',name),tools):self.assertEqual(args,[{'cmd':'fixture'}]);self.assertFalse(p.has_rejected_tool_calls)
 def test_unknown_schema_does_not_infer_unwrap(self):
  original='{"arguments":"{\\"cmd\\":\\"fixture\\"}"}'
  self.assertEqual(validate_or_unwrap(original,None),original)
  tools=[{'type':'function','name':'exec_command','parameters':None}]
  for args,p in self.paths(xml('{"cmd":"fixture"}'),tools):self.assertEqual(args,[{'arguments':'{"cmd":"fixture"}'}])
 def test_unknown_name_not_authorized_by_wrapper(self):
  for args,p in self.paths(xml('{"cmd":"fixture"}','write_file')):self.assertFalse(args);self.assertTrue(p.has_rejected_tool_calls)
 def test_implicit_thinking_call_uses_same_normalizer(self):
  for args,p in self.paths('Fixture.\n'+xml('{"cmd":"fixture"}'),thinking=True):self.assertEqual(args,[{'cmd':'fixture'}]);self.assertFalse(p.has_rejected_tool_calls)
 def test_repeated_full_reconstruction_is_idempotent(self):
  req,p,prompt=f.make(TOOLS,False);text=xml('{"cmd":"fixture"}')
  for _ in range(3):
   result=p.parse(text,req,enable_auto_tools=True,model_output_token_ids=f.ids(text) )
   self.assertEqual(json.loads(result[2][0].arguments),{'cmd':'fixture'})
 def test_duplicate_json_keys_and_nonfinite_not_losslessly_repaired(self):
  for text in ['{"cmd":"a","cmd":"b"}','{"cmd":"fixture","count":NaN}']:
   with self.assertRaises(InvalidToolArguments):validate_or_unwrap(json.dumps({'arguments':text}),SCHEMA)
 def test_schema_local_ref_and_nested_types_checked(self):
  schema={'$defs':{'cmd':{'type':'string'}},'type':'object','properties':{'cmd':{'$ref':'#/$defs/cmd'}},'required':['cmd'],'additionalProperties':False}
  self.assertEqual(json.loads(validate_or_unwrap('{"arguments":{"cmd":"fixture"}}',schema)),{'cmd':'fixture'})
  with self.assertRaises(InvalidToolArguments):validate_or_unwrap('{"arguments":{"cmd":5}}',schema)
 def test_remote_ref_never_retrieved(self):
  schema={'$ref':'https://example.invalid/do-not-fetch.json'}
  with self.assertRaises(InvalidToolArguments):validate_or_unwrap('{"cmd":"fixture"}',schema)
 def test_invalid_supplied_schema_fails_safely(self):
  with self.assertRaises(InvalidToolArguments):validate_or_unwrap('{"cmd":"fixture"}',{'type':'not-a-type'})
 def test_plain_direct_command_unchanged(self):
  text=f.call(value='printf fixture')
  for args,p in self.paths(text):self.assertEqual(args,[{'cmd':'printf fixture'}]);self.assertFalse(p.has_rejected_tool_calls)

 def test_unknown_schema_dialect_fails_without_guessing(self):
  with self.assertRaises(InvalidToolArguments):validate_or_unwrap('{"cmd":"fixture"}',{'$schema':'https://example.invalid/unknown-dialect','type':'object'})
