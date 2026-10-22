#!/usr/bin/env python3
"""Live long-context retrieval, greedy replay and prefix-cache probe."""
import argparse
import json
import random
import time
import urllib.request


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--base-url', default='http://localhost:18300')
    cli.add_argument('--tokens', type=int, default=499000)
    cli.add_argument('--output', required=True)
    args = cli.parse_args()
    def post(path, body):
        request = urllib.request.Request(args.base_url + path, json.dumps(body).encode(), {'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)
    model = json.load(urllib.request.urlopen(args.base_url + '/v1/models'))['data'][0]['id']
    rng = random.Random(32021)
    vocabulary = 'ledger invoice payroll contract clause annex schedule amount vendor total tax paid harbour lantern quartz meadow copper signal'.split()
    words = [rng.choice(vocabulary) + (str(rng.randrange(10000)) if rng.random() < .15 else '') for _ in range(args.tokens)]
    def messages(n):
        half = n//2
        text = ' '.join(words[:half]) + '\nIMPORTANT NOTE: the secret vault code is 583917. Remember it.\n' + ' '.join(words[half:n])
        return [{'role':'system','content':'Answer with the number only.'}, {'role':'user','content':text+'\nWhat is the secret vault code? Reply with the number only.'}]
    n = int(args.tokens/1.55)
    template = {'reasoning_effort':'low'}
    for _ in range(5):
        msgs = messages(n)
        count = post('/tokenize', {'model':model,'messages':msgs,'chat_template_kwargs':template})['count']
        if abs(count-args.tokens) <= 50:
            break
        n = max(1, int(n*(args.tokens-10)/count))
    assert count <= args.tokens + 50, count
    report = {'model_inference':True, 'prompt_tokens_tokenize':count, 'runs':[]}
    outputs = []
    for repeat in range(2):
        started = time.monotonic()
        response = post('/v1/chat/completions', {'model':model,'messages':msgs,'max_tokens':128,'temperature':0,'chat_template_kwargs':template})
        choice = response['choices'][0]
        content = choice['message'].get('content') or ''
        usage = response['usage']
        assert content.strip() == '583917', {'content':content, 'finish_reason':choice['finish_reason']}
        outputs.append(content)
        report['runs'].append({'repeat':repeat, 'elapsed_s':time.monotonic()-started, 'usage':usage, 'content':content, 'finish_reason':choice['finish_reason']})
    assert outputs[0] == outputs[1]
    cached = (report['runs'][1]['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)
    assert cached > count*.9, {'cached_tokens':cached, 'count':count}
    report['deterministic'] = True
    report['prefix_cache_hit'] = True
    with open(args.output,'w') as out:
        json.dump(report,out,indent=2)
        out.write('\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
