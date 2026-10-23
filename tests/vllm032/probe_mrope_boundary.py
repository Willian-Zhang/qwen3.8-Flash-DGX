#!/usr/bin/env python3
"""Live issue-60 retrieval and health checks across the native M-RoPE boundary.

Use an isolated candidate: an unfixed engine can die on this probe. No tools are
offered or executed. Reports contain hashes and short answers, not long prompts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base-url", default="http://127.0.0.1:18360")
    cli.add_argument("--sizes", default="260000,263000,336000,499000")
    cli.add_argument("--repeats", type=int, default=2)
    cli.add_argument("--timeout", type=int, default=900)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    if not sizes or min(sizes) < 100 or max(sizes) > 499000 or args.repeats < 1:
        cli.error("sizes must be 100..499000 and repeats must be positive")
    report = {"model_inference": True, "tools_executed": 0,
              "native_boundary": 262144, "cases": [], "all_passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)

    def get(path):
        with urllib.request.urlopen(args.base_url + path, timeout=10) as response:
            return response.status, response.read()

    def post(path, body):
        data = json.dumps(body).encode()
        request = urllib.request.Request(args.base_url + path, data,
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    def healthy():
        try:
            return get("/health")[0] == 200
        except (OSError, urllib.error.HTTPError):
            return False

    try:
        model = json.loads(get("/v1/models")[1])["data"][0]["id"]
        report["model"] = model
        template = {"reasoning_effort": "low"}
        for target in sizes:
            case = {"requested_prompt_tokens": target, "runs": []}
            report["cases"].append(case)
            save()
            rng = random.Random(60000 + target)
            vocabulary = "ledger invoice payroll contract clause annex schedule amount vendor total tax paid harbour lantern quartz meadow copper signal".split()
            words = [rng.choice(vocabulary) + (str(rng.randrange(10000))
                     if rng.random() < .15 else "") for _ in range(target)]
            expected = str(100000 + (target * 31337) % 900000)

            def messages(n):
                half = n // 2
                text = (f"Boundary fixture issue60-20261022-{target}.\n"
                        + " ".join(words[:half])
                        + f"\nIMPORTANT NOTE: the secret vault code is {expected}. Remember it.\n"
                        + " ".join(words[half:n])
                        + "\nWhat is the secret vault code? Reply with the number only.")
                return [{"role": "system", "content": "Answer with the number only."},
                        {"role": "user", "content": text}]

            n = int(target / 1.55)
            for _ in range(10):
                msgs = messages(n)
                count = post("/tokenize", {"model": model, "messages": msgs,
                             "chat_template_kwargs": template})["count"]
                if abs(count - target) <= 8:
                    break
                n = max(1, min(len(words), int(n * target / count)))
            if abs(count - target) > 32:
                raise AssertionError(f"Token calibration: {target=} {count=}")
            case.update(prompt_tokens_tokenize=count, expected=expected,
                        request_sha256=hashlib.sha256(json.dumps(msgs).encode()).hexdigest())
            for repeat in range(args.repeats):
                run = {"repeat": repeat}
                case["runs"].append(run)
                save()
                started = time.monotonic()
                try:
                    response = post("/v1/chat/completions", {
                        "model": model, "messages": msgs, "max_tokens": 256,
                        "temperature": 0, "chat_template_kwargs": template,
                    })
                    choice = response["choices"][0]
                    content = choice["message"].get("content") or ""
                    run.update(usage=response["usage"], content=content,
                               finish_reason=choice["finish_reason"],
                               answer_correct=content.strip() == expected)
                    if repeat:
                        cached = (response["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                        run["prefix_cache_reused"] = cached > count * .9
                except Exception as exc:
                    run["error"] = type(exc).__name__
                    if isinstance(exc, urllib.error.HTTPError):
                        run["http_status"] = exc.code
                        run["error_body"] = exc.read(2048).decode(errors="replace")
                    raise
                finally:
                    run["elapsed_s"] = time.monotonic() - started
                    run["healthy_after"] = healthy()
                    save()
                if not run["healthy_after"]:
                    raise RuntimeError("Engine unhealthy after long request")
                check = post("/v1/chat/completions", {
                    "model": model, "messages": [{"role": "user", "content": "Reply READY."}],
                    "max_tokens": 32, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                })
                run["short_request_after"] = bool(check.get("choices"))
                save()
                print(json.dumps({"tokens": count, **run}), flush=True)
        report["all_passed"] = all(
            row.get("healthy_after") and row.get("answer_correct")
            and row.get("short_request_after")
            and (not row["repeat"] or row.get("prefix_cache_reused"))
            for case in report["cases"] for row in case["runs"]
        )
    except Exception as exc:
        report["error"] = type(exc).__name__ + ": " + str(exc)
    finally:
        report["healthy_at_end"] = healthy()
        save()
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
