import asyncio,json,sys
from pathlib import Path
QA = Path(__file__).resolve().parents[1] / 'issue31_safe_output'
sys.path[:0] = [str(QA), str(QA / 'tests')]
import producer_tests as responses
from chat_fixture import run,full
value={'cmd':'printf fixture'}
call='<tool_call>\n<function=exec_command>\n<parameter=arguments>'+json.dumps(value)+'</parameter>\n</function>\n</tool_call>'
text='Fixture.</think>'+call
async def main():
 result={'model_inference':False,'source_overlay':False,'cases':{}}
 for size in (1,7,100000):
  events,_=await responses.actual_stream(text,'stop',size)
  last=events[-1];result['cases']['responses-stream-'+str(size)]={'terminal':last.type,'calls':[json.loads(i.arguments) for i in last.response.output if i.type=='function_call']}
  result['cases']['chat-stream-'+str(size)]=await run(text,True,size)
 response,_=await responses.actual_full(text,'stop')
 result['cases']['responses-full']={'status':getattr(response,'status',None),'calls':[json.loads(i.arguments) for i in getattr(response,'output',[]) if i.type=='function_call']}
 response=await full(text,True)
 result['cases']['chat-full']={'calls':[json.loads(i.function.arguments) for i in response.choices[0].message.tool_calls or []]}
 print(json.dumps(result,indent=2))
asyncio.run(main())
