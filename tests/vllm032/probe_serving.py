#!/usr/bin/env python3
"""Live Chat/Responses tool protocol probe. Proposed tools are never executed."""
import argparse
import json
import time
import urllib.request


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base-url', default='http://localhost:18300')
    cli.add_argument('--output', required=True)
    args = cli.parse_args()
    def post(path, body):
        request = urllib.request.Request(args.base_url + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=180) as response:
            if not body.get('stream'):
                return json.load(response)
            return [json.loads(line[6:]) for line in response
                    if line.startswith(b'data: ') and line.strip() != b'data: [DONE]']
    model = json.load(urllib.request.urlopen(args.base_url + '/v1/models'))['data'][0]['id']
    tool = {'type': 'function', 'name': 'add_numbers', 'description': 'Add the two integers.',
            'parameters': {'type': 'object', 'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                           'required': ['a', 'b'], 'additionalProperties': False}}
    prompt = 'Call add_numbers with a=2 and b=3. Use the tool, then report its result.'
    report = {'model': model, 'model_inference': True, 'tools_executed': 0, 'checks': {}, 'all_passed': False}
    started = time.monotonic()
    messages = [{'role': 'user', 'content': prompt}]
    body = {'model': model, 'messages': messages, 'tools': [{'type': 'function', 'function': {k:v for k,v in tool.items() if k != 'type'}}],
            'temperature': 0, 'max_tokens': 512, 'chat_template_kwargs': {'reasoning_effort': 'low'}}
    chat = post('/v1/chat/completions', body)
    assistant = chat['choices'][0]['message']
    calls = assistant.get('tool_calls') or []
    assert len(calls) == 1, chat
    assert calls[0]['function']['name'] == 'add_numbers', calls
    assert json.loads(calls[0]['function']['arguments']) == {'a': 2, 'b': 3}, calls
    report['checks']['chat_tool_arguments'] = True
    # A fixed synthetic result exercises continuation without executing a tool.
    continuation = post('/v1/chat/completions', {**body, 'messages': messages + [assistant, {'role':'tool','tool_call_id':calls[0]['id'],'content':'5'}]})
    assert '5' in (continuation['choices'][0]['message'].get('content') or ''), continuation
    report['checks']['chat_tool_round_trip'] = True
    for stream in (False, True):
        response = post('/v1/responses', {'model':model, 'input':prompt, 'tools':[tool], 'temperature':0,
                        'max_output_tokens':512, 'stream':stream, 'store':False,
                        'chat_template_kwargs':{'reasoning_effort':'low'}})
        if stream:
            assert response[-1]['type'] == 'response.completed', response[-1]
            added = [e['item'] for e in response if e['type']=='response.output_item.added' and e['item']['type']=='function_call']
            response = response[-1]['response']
        output = [i for i in response['output'] if i['type']=='function_call']
        assert len(output)==1 and output[0]['name']=='add_numbers', response
        assert json.loads(output[0]['arguments'])=={'a':2,'b':3}, output
        if stream:
            identities = {'added': [(i['id'], i['call_id']) for i in added],
                          'completed': [(i['id'], i['call_id']) for i in output]}
            report['responses_stream_identities'] = identities
            with open(args.output, 'w') as out:
                json.dump(report, out, indent=2)
                out.write('\n')
            assert identities['added'] == identities['completed'], identities
        report['checks']['responses_stream' if stream else 'responses_full'] = True
    report['all_passed'] = True
    report['elapsed_s'] = time.monotonic() - started
    with open(args.output, 'w') as out:
        json.dump(report, out, indent=2)
        out.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
