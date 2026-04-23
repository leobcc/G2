import json

with open('results/latest_results.json') as f:
    d = json.load(f)

print(f"Deals processed: {len(d['results'])}")
for r in d['results']:
    ev = r['evaluation']
    delta = ev.get('composite_score_delta', 0)
    verdict = ev.get('verdict', '?')
    llm = ev.get('llm_judge_score', 'N/A')
    rouge = ev.get('rouge_l', 'N/A')
    print(f"  rank={r['rank']} id={r['deal_id']} verdict={verdict} score_delta=+{delta:.1f} llm_judge={llm} rouge_l={rouge}")

print()
print("Summary:", d.get('summary', {}))
