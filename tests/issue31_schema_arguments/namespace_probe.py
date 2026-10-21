import json
import test_parser_safety as f
SCHEMA={'type':'object','properties':{'cmd':{'type':'string'}},'required':['cmd'],'additionalProperties':False}
tools=[{'type':'namespace','name':'functions_group','description':'Fixture namespace','tools':[{'type':'function','name':'run','description':'fixture','parameters':SCHEMA}]}]
rows={}
for name in ('functions_group.run','functions_group__run','functions.functions_group.run'):
 text='<tool_call><function='+name+'><parameter=arguments>{"cmd":"fixture"}</parameter></function></tool_call>'
 result,p=f.full(text,tools=tools)
 rows[name]={'calls':[{'name':i.name,'args':json.loads(i.arguments)} for i in result[2] or []],'rejected':p.has_rejected_tool_calls}
print(json.dumps({'model_inference':False,'source_overlay':False,'cases':rows},indent=2))
