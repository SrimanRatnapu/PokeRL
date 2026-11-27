"""Pick the blend weight with the best mean win rate (vs lookahead and vs champion) from logs/e10-beta-sweep.log.

beta 0 (plain policy) is excluded; ties go to the smaller beta. Prints the beta, or 2 if the log has no rows.
"""
import re
import sys
from collections import defaultdict

rates = defaultdict(list)
for line in open(sys.argv[1] if len(sys.argv) > 1 else "logs/e10-beta-sweep.log"):
    m = re.match(r"beta=([\d.]+) vs \w+:.*?(\d+)/(\d+)", line)
    if m:
        rates[float(m.group(1))].append(int(m.group(2)) / int(m.group(3)))
full = {b: sum(r) / len(r) for b, r in rates.items() if len(r) == 2 and b > 0}
print(f"{max(sorted(full), key=lambda b: full[b]):g}" if full else "2")
