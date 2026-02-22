import os, json, glob
exp_dir = r"results\3ARM-VS-4ARM_20260219_022444"
runs = glob.glob(os.path.join(exp_dir, "run-GA-LLAMEA-3arm*"))
for r in runs:
    lf = os.path.join(r, "log.jsonl")
    try:
        with open(lf) as f: lines = f.readlines()
        valid = 0
        gen_max = 0
        for i, l in enumerate(lines):
            try:
                j = json.loads(l.strip())
                if j.get("fitness") not in ["-inf", None]:
                    valid += 1
                    gen_max = i
            except Exception: pass
        mstr = "run="+os.path.basename(r)+" lines="+str(len(lines))+" valid="+str(valid)+" max="+str(gen_max)
        with open("out.txt", "a") as f2: f2.write(mstr+"\n")
    except Exception as e: pass
