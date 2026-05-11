# Candidate Queries (50) — Pick 30 for the Eval Set

**Created:** 2026-05-09
**Source corpus:** `data/trend_signals.jsonl` — 400 translated Chinese news titles across 11 platforms.

## Corpus theme distribution (from analysis)

| Theme | % of corpus | Notes |
|---|---|---|
| Geopolitics | 17% | China, Iran, Hormuz, US-China trade, Trump, war |
| Economy / finance | 11% | Stocks, yuan, May Day spending, payrolls, real estate |
| Sports | 6% | World Cup broadcasting, table tennis, FIFA, esports (Faker) |
| Entertainment | 5% | Disney, anime, celebrities, FanMakers, music |
| Health / medical | 3% | **Hantavirus outbreak (very concentrated topic)**, food poisoning, hospitals |
| Family / parenting | 3% | Mom questionnaires, raising kids, marriage, divorce |
| Education | 2% | School staff, university, exams |
| Tech / AI | 1% | Gen Z startups, AI for chemistry |
| Real estate | 1% | Shanghai sales, May Day home buying |
| Social issues | 1% | Crime, fraud, sentencing |

The corpus is **dominated by geopolitics, economy, and sports** — design queries accordingly. Family/parenting is a small slice (the original niche angle of the source data).

---

## How to use this file

1. Read all 50 candidates below
2. Pick 30 that match the coverage targets in `labeling_protocol.md`:
   - 8 broad topic
   - 10 specific entity
   - 5 multi-aspect
   - 3 negative test
   - 4 family/parenting niche
3. Copy chosen queries into `eval_results/eval_set.jsonl` (replace TODOs)
4. Label with `relevant_doc_ids` per the protocol

Each candidate has:
- **Query text** (use as-is or rewrite)
- **Type** (broad / specific / multi-aspect / negative / family-niche)
- **Predicted relevant doc count** (rough — actual count is what you label)
- **Why it tests something useful** (brief)

---

## A. Broad topic queries (target: 8 of 8 listed)

These have many relevant docs — test recall@10.

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| A1 | what trends are about US-China geopolitics or trade | broad | 15-25 | Largest theme; tests if retrieval surfaces the dominant topic cleanly |
| A2 | what trending topics relate to Chinese economic indicators | broad | 8-15 | Macro-finance; common recruiter use case |
| A3 | what news involves Iran or the Middle East | broad | 8-12 | Concentrated in the corpus (Hormuz crisis dominates) |
| A4 | what sports stories are trending right now | broad | 15-20 | World Cup + table tennis dominate; tests sports cluster |
| A5 | what entertainment news is being discussed online | broad | 12-18 | Disney, anime, celebrity content |
| A6 | what trends involve stock markets and trading | broad | 6-10 | Tests financial-news subtopic |
| A7 | which trends are about Chinese government policy or officials | broad | 8-12 | Domestic politics — distinct from US-China |
| A8 | what topics are most viral on Chinese social media this week | broad | 30+ | Catch-all — tests if any specific signal beats noise |

---

## B. Specific entity queries (target: 10 of 12 listed)

These have few but exact matches — test precision@5.

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| B1 | what is the hantavirus outbreak situation | specific | 8-11 | Very concentrated topic (11 mentions) — test entity recall |
| B2 | tell me about the Hormuz strait military situation | specific | 5-7 | Specific geographic event |
| B3 | what is happening with Trump and China relations | specific | 4-6 | Named entity + topic |
| B4 | who is Faker and what is happening in his career | specific | 1-3 | Esports celebrity — tests narrow named-entity retrieval |
| B5 | what news mentions the World Cup broadcast rights | specific | 4-6 | Sub-event within the broader sports cluster |
| B6 | what is the China men's table tennis team doing | specific | 3-5 | Specific sports team entity |
| B7 | what are people saying about the Jiajie sandwich incident | specific | 1-3 | Niche scandal — tests rare-entity retrieval |
| B8 | tell me about the WSJ Morning Radio segments | specific | 3-5 | Recurring source pattern |
| B9 | what is the Real Madrid locker room news | specific | 1-2 | Sports celebrity gossip — narrow |
| B10 | what is happening in Shanghai real estate | specific | 2-4 | Entity + region |
| B11 | what is the Lao A horror story trending about | specific | 1 | Single-doc test |
| B12 | what news mentions Wang Yi the Chinese foreign minister | specific | 2-4 | Named individual |

---

## C. Multi-aspect queries (target: 5 of 8 listed)

Two-concept queries that test compositional vector similarity.

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| C1 | what is happening with Chinese tariffs and US trade response | multi-aspect | 3-5 | Trade + tariff specifically |
| C2 | what celebrity news involves financial scandal or money | multi-aspect | 2-4 | Two themes intersection |
| C3 | what trends discuss housing prices during the May Day holiday | multi-aspect | 2-3 | Real estate + holiday timing |
| C4 | what news involves both technology and Chinese startups | multi-aspect | 1-3 | Tech + China |
| C5 | what stories are about food poisoning or restaurant safety | multi-aspect | 2-4 | Food + safety |
| C6 | what trends discuss government corruption or sentencing | multi-aspect | 3-5 | Crime + government |
| C7 | what topics combine sports and Chinese national pride | multi-aspect | 4-6 | Sports + nationalism |
| C8 | what is happening with stock markets and oil prices | multi-aspect | 3-5 | Two financial subtopics |

---

## D. Negative test queries (target: 3 of 5 listed)

Expect 0 relevant — tests that retrieval doesn't false-match. The LLM should refuse to answer when the system is wired up.

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| D1 | how do I implement a binary search tree in Python | negative | 0 | Programming tutorial — corpus is news only |
| D2 | what are the best practices for FastAPI authentication | negative | 0 | Tech docs — not in corpus |
| D3 | who won the 2024 Nobel Prize in physics | negative | 0 | Specific factual query outside corpus topic/timeframe |
| D4 | what is the recipe for chocolate chip cookies | negative | 0 | Recipe content — corpus has cooking trends but no recipes |
| D5 | how do I treat my child's ear infection | negative | 0-1 | Medical advice — corpus has health-event trends, not clinical |

---

## E. Family / parenting niche queries (target: 4 of 8 listed)

The original niche angle of the source dataset. Small corpus slice (3%) so retrieval is a real challenge here.

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| E1 | what trending topics are about parenting or raising children | family-niche | 3-6 | Direct hit on the niche family-content use case |
| E2 | what news discusses marriage or divorce | family-niche | 3-5 | Family relationship subtopic |
| E3 | what stories mention mothers or motherhood | family-niche | 2-4 | "Mom" specifically |
| E4 | what trends are about saving money for the family | family-niche | 1-3 | Stretch — corpus has money + family but not specifically family-savings |
| E5 | what news covers school or children's education | family-niche | 3-5 | Education subtopic |
| E6 | what topics discuss family values or traditional households | family-niche | 1-3 | Cultural-family angle |
| E7 | what trends involve weddings or relationships | family-niche | 2-4 | Adjacent to family theme |
| E8 | what stories discuss family budgeting or household finances | family-niche | 1-2 | The hardest niche query — corpus has personal finance but not family-budgeting how-to |

---

## F. Bonus (pick if you want extra coverage — optional)

| # | Query | Type | Predicted relevant | Why useful |
|---|---|---|---|---|
| F1 | what is being said about Disney's pricing or business decisions | specific | 2-3 | Adjacent to entertainment theme |
| F2 | what social media platforms are mentioned in trending news | meta | 0-2 | Tests if platform names leak into titles |
| F3 | what trends involve Russian or Ukrainian war news | specific | 1-3 | Geopolitics subset |
| F4 | what news involves Brazil or Latin America | specific | 1-2 | Geographic outlier |
| F5 | what stories discuss internet scams or online fraud | specific | 2-4 | Crime subtopic |
| F6 | what topics involve Japanese culture or Japan | specific | 4-6 | Geographic sub-cluster |
| F7 | what news involves AI startups or artificial intelligence | specific | 1-3 | Tech-AI corner |

---

## Recommended pick for the 30-query eval set

If you want a starter recommendation, here's a balanced 30 from the above:

**Broad (8):** A1, A2, A3, A4, A5, A6, A7, A8
**Specific (10):** B1, B2, B3, B4, B5, B6, B7, B9, B10, B12
**Multi-aspect (5):** C1, C3, C5, C6, C8
**Negative (3):** D1, D2, D5
**Family-niche (4):** E1, E2, E3, E8

That's 30 queries spanning all categories, weighted toward the corpus's actual topic distribution. Each will take ~2 minutes to label = ~60 minutes total.

If you'd rather pick differently — go for it. The categories matter more than the specific queries.

---

## After picking

1. Copy your chosen 30 queries into `eval_results/eval_set.jsonl` (overwrite the TODO entries)
2. Use the labeling workflow in `labeling_protocol.md` to fill `relevant_doc_ids`
3. When done, run `src/eval/retrieval.py` (lands Day 4 implementation) to compute precision@k, recall@k, MRR
4. Commit `eval_set.jsonl` + `baseline.json`
