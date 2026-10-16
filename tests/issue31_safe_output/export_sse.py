import asyncio,json,os,shutil,sys,tempfile
from pathlib import Path
with tempfile.TemporaryDirectory(prefix='issue31-sse-export-') as tmp:
 d=Path(tmp)/'tests';shutil.copytree('/qa/tests',d);sys.path.insert(0,str(d))
 import producer_tests as p
 cases={
  'length_visible':('<think>Fixture.</think>Partial','length',True),
  'length_reasoning_hidden':('<think>Fixture only.','length',False),
  'stop_visible':('<think>Fixture.</think>Done.','stop',True),
  'stop_empty':('','stop',False),
 }
 report={'source_overlay':False,'model_inference':False,'cases':{}}
 for name,(text,finish,include) in cases.items():
  events,_=asyncio.run(p.actual_stream(text,finish,7,include))
  report['cases'][name]=[event.model_dump(mode='json',by_alias=True) for event in events]
 Path('/out/'+os.environ['QA_LABEL']+'-sse-fixtures.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:{'terminal':v[-1]['type'],'status':v[-1].get('response',{}).get('status')} for k,v in report['cases'].items()},indent=2))
