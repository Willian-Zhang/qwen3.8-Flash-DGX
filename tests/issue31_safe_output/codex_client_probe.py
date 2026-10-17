import argparse,http.server,json,os,pathlib,socketserver,subprocess,tempfile,threading,time
ap=argparse.ArgumentParser();ap.add_argument('fixtures',type=pathlib.Path);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--require-correct',action='store_true');args=ap.parse_args()
data=json.loads(args.fixtures.read_text());report={'client':subprocess.check_output(['codex','--version'],text=True,stderr=subprocess.DEVNULL).strip(),'model_inference':False,'fixtures':str(args.fixtures),'cases':{}}
for name,events in data['cases'].items():
 requests=[]
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*a):pass
  def do_GET(self):
   body=json.dumps({'object':'list','data':[{'id':'qwen3.8-flash-next','object':'model','created':0,'owned_by':'local-fixture'}]}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_POST(self):
   raw=self.rfile.read(int(self.headers.get('Content-Length',0)))
   # Retain only structural request counts; never headers or user credentials.
   request=json.loads(raw);requests.append({'path':self.path,'tools_offered':len(request.get('tools',[]))})
   body=b''.join(('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n').encode() for e in events)
   self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(body)));self.end_headers()
   try:self.wfile.write(body)
   except (BrokenPipeError,ConnectionResetError):pass
 with socketserver.TCPServer(('127.0.0.1',0),Handler) as server,tempfile.TemporaryDirectory(prefix='issue31-codex-fixture-') as project:
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  base=f'http://127.0.0.1:{server.server_address[1]}/v1'
  command=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','--json','--sandbox','read-only','-C',project,'-m','qwen3.8-flash-next']
  config={'model_provider':'issue31_fixture','model_providers.issue31_fixture.name':'Local synthetic Responses fixture','model_providers.issue31_fixture.base_url':base,'model_providers.issue31_fixture.wire_api':'responses','model_providers.issue31_fixture.requires_openai_auth':False,'model_providers.issue31_fixture.supports_websockets':False,'model_providers.issue31_fixture.request_max_retries':0,'model_providers.issue31_fixture.stream_max_retries':0}
  for key,value in config.items():command+=['-c',key+'='+json.dumps(value)]
  command+=['Reply exactly OK without using tools.']
  started=time.monotonic()
  try:
   proc=subprocess.run(command,capture_output=True,text=True,timeout=60)
   rows=[]
   for line in proc.stdout.splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
   errors=[row for row in rows if row.get('type') in ('error','turn.failed')]
   report['cases'][name]={'exit_code':proc.returncode,'wall_s':time.monotonic()-started,'requests':len(requests),'events':[row.get('type') for row in rows],'errors':errors,'stderr_tail':proc.stderr[-1600:],'tool_executions':sum(row.get('type')=='item.completed' and row.get('item',{}).get('type') in ('command_execution','mcp_tool_call','web_search','file_change','collab_tool_call') for row in rows)}
  except subprocess.TimeoutExpired:report['cases'][name]={'timeout':True,'requests':len(requests)}
  finally:server.shutdown()
 args.output.write_text(json.dumps(report,indent=2)+'\n')
 print(name,json.dumps(report['cases'][name]),flush=True)

for name, case in report['cases'].items():
 terminal=data['cases'][name][-1]
 expected_failure=name in ('length_visible','length_reasoning_hidden','stop_empty') or terminal.get('response',{}).get('status') in ('incomplete','failed')
 case['expected_exit_code']=1 if expected_failure else 0
 case['correct_client_outcome']=case.get('exit_code')==case['expected_exit_code'] and case.get('requests')==1 and case.get('tool_executions')==0
report['passed']=all(c.get('correct_client_outcome') for c in report['cases'].values())
args.output.write_text(json.dumps(report,indent=2)+'\n')
if args.require_correct and not report['passed']:raise SystemExit(1)
