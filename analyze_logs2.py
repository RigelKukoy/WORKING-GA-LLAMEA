import json
import os

base = r"results\COMPARISON-PROMPTS_20260307_121534"

for d in sorted(os.listdir(base)):
    dirpath = os.path.join(base, d)
    logfile = os.path.join(dirpath, "log.jsonl")
    if not os.path.isfile(logfile):
        continue
    print(f"\n=== {d} ===")
    with open(logfile, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            gen = entry.get("generation", "?")
            fitness = entry.get("fitness", "?")
            name = entry.get("name", "?")
            error = str(entry.get("error", ""))[:250]
            feedback = str(entry.get("feedback", ""))[:250]
            print(f"  [{i}] gen={gen} fitness={fitness} name={name}")
            if error:
                print(f"       error: {error}")
            elif feedback:
                print(f"       feedback: {feedback}")
