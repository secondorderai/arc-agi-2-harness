# V2 public evaluation was dominated by model timeouts and fallback selection

Generated `2026-09-03T02:11:12.510435+00:00` from the frozen evaluation workspace. This is a post-hoc diagnostic: labels were used only after submission freeze.

## Executive finding

V2 solved **3/120 tasks (2.50%)** and **3/167 outputs (1.80% Pass@2)**. The run did not meaningfully test the intended resynthesis strategy across the full set: **115/121 model requests (95.04%) timed out**, while only **3/120 tasks** obtained any guard-passing candidate.

The three solved tasks were all backed by selected guard-passing programs (3 guarded; 0 unguarded). The other 117 tasks failed at exact-match scoring.

## Where the pipeline failed

| Outcome / primary operational diagnosis | Tasks | Share |
|---|---:|---:|
| Correct: guard-passing program | 3 | 2.50% |
| Model timed out; fallback failed | 115 | 95.83% |
| Synthesis failed guards; fallback failed | 2 | 1.67% |

The diagnosis is assigned from the observable pipeline state, not inferred ARC semantics. A timeout diagnosis means the task had no selected guard-passing program and its model turn exceeded the configured 360-second limit before a usable response was ingested.

## Evidence by stage

- Retrieval evaluated **6,388** bank programs, but only **1** passed every evaluation demonstration guard.
- Luna produced only **5 parsed candidates** across **4 tasks**; **2** passed all guards.
- The scheduler recorded **5 completed**, **115 timed-out**, and **1 failed** requests.
- At least one fallback attempt was used on **18 tasks**.
- The post-hoc candidate oracle finds **3/167 outputs** and **3/120 tasks** represented somewhere in the stored candidate pool. This is an analysis-only upper bound because it uses labels.
- **0 failed tasks** had a fully correct stored candidate that was not selected. These are concrete selection-or-guard-ranking opportunities; all other failures require better candidate generation or completed model responses.

## Interpretation

The strongest supported conclusion is operational: V2's 2.50% score is primarily a **coverage failure**. The six-minute model-turn ceiling combined with two LLM slots and a six-hour first-model stage left nearly every task without a completed resynthesis. The direct ranker then supplied high-scoring partial programs, but demonstration guards admitted only one of them. Consequently, the submitted second attempt was often an input-copy or zero-grid fallback rather than an independently synthesized solution.

This result does not establish that xhigh Luna resynthesis is intrinsically ineffective on ARC-AGI-2. It establishes that this harness configuration rarely captured a completed Luna result within its per-turn and stage budgets.

## Recommended V3 gates

1. Run a 10-task held-out smoke test and require at least 9 model responses to be ingested before the turn deadline. Shorten prompts/output limits or reduce reasoning effort first; increasing the timeout without increasing throughput would reduce 120-task coverage.
2. Treat timed-out background work as resumable instead of terminal: preserve the response/thread identifier, poll it later, and ingest late completion exactly once when the global budget permits.
3. Rework retrieval around task-family and behavior features. Training top-32 recall did not translate into public demonstration compatibility: only 1 of 120 tasks found a guard-passing direct program in the scanned pool.
4. Use the post-hoc oracle only to test candidate selection on held-out training tasks. Never use public labels in the deployable selector.
5. Repeat the frozen 12-hour protocol in a new workspace after the smoke gates pass; preserve this run as the V2 baseline.

## Per-task audit (all 120 tasks)

`Oracle` is the number of labelled outputs matched by any stored candidate, whether or not that candidate passed guards. `Cell` is the mean best cell accuracy across the two submitted attempts.

| Task | Result | Primary diagnosis | Outputs | Correct | Direct guarded | Synth guarded | Requests C/T/F | Sources | Cell | Oracle |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| `0934a4d8` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `135a2760` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 98.93% | 0 |
| `136b0064` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `13e47133` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 12.61% | 0 |
| `142ca369` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 66.00% | 0 |
| `16b78196` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 89.22% | 0 |
| `16de56c4` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 78.04% | 0 |
| `1818057f` | correct | Correct: guard-passing program | 1 | 1 | 0 | 1 | 1/0/0 | luna / direct | 100.00% | 1 |
| `195c6913` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 89.39% | 0 |
| `1ae2feb7` | failed | Model timed out; unguarded fallback failed | 3 | 0 | 0 | 0 | 0/1/0 | direct / direct | 80.33% | 0 |
| `20270e3b` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `20a9e565` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `21897d95` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 35.92% | 0 |
| `221dfab4` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 78.70% | 0 |
| `247ef758` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 86.19% | 0 |
| `269e22fb` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 31.37% | 0 |
| `271d71e2` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 63.91% | 0 |
| `28a6681f` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 95.00% | 0 |
| `291dc1e1` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `2b83f449` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 85.76% | 0 |
| `2ba387bc` | failed | Synthesis returned but failed guards | 1 | 0 | 0 | 0 | 1/0/0 | direct / direct | 0.00% | 0 |
| `2c181942` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 92.95% | 0 |
| `2d0172a1` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `31f7f899` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 89.62% | 0 |
| `332f06d7` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 92.00% | 0 |
| `35ab12c3` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 84.13% | 0 |
| `36a08778` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 81.13% | 0 |
| `38007db0` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `3a25b0d8` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 42.53% | 0 |
| `3dc255db` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 89.74% | 0 |
| `3e6067c3` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 96.28% | 0 |
| `409aa875` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 96.48% | 0 |
| `446ef5d2` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 69.73% | 0 |
| `45a5af55` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `4a21e3da` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.25% | 0 |
| `4c3d4a41` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.50% | 0 |
| `4c416de3` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 94.02% | 0 |
| `4c7dc4dd` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `4e34c42c` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `53fb4810` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.78% | 0 |
| `5545f144` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 91.11% | 0 |
| `581f7754` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.96% | 0 |
| `58490d8a` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 90.91% | 0 |
| `58f5dbd5` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `5961cc34` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.17% | 0 |
| `5dbc8537` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `62593bfd` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 88.10% | 0 |
| `64efde09` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 85.73% | 0 |
| `65b59efc` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `67e490f4` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 68.57% | 0 |
| `6e453dd6` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 77.33% | 0 |
| `6e4f6532` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 85.43% | 0 |
| `6ffbe589` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `71e489b6` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.24% | 0 |
| `7491f3cf` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.14% | 0 |
| `7666fa5d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 79.30% | 0 |
| `78332cb0` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 44.78% | 0 |
| `7b0280bc` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.78% | 0 |
| `7b3084d4` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `7b5033c1` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `7b80bb43` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 97.16% | 0 |
| `7c66cb00` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 88.33% | 0 |
| `7ed72f31` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.27% | 0 |
| `800d221b` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 84.00% | 0 |
| `80a900e0` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 90.26% | 0 |
| `8698868d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `88bcf3b4` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 95.05% | 0 |
| `88e364bc` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 98.62% | 0 |
| `89565ca0` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `898e7135` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 11.89% | 0 |
| `8b7bacbf` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 94.19% | 0 |
| `8b9c3697` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 89.53% | 0 |
| `8e5c0c38` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 97.11% | 0 |
| `8f215267` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 92.11% | 0 |
| `8f3a5a89` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 67.36% | 0 |
| `9385bd28` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 78.74% | 0 |
| `97d7923e` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 94.00% | 0 |
| `981571dc` | correct | Correct: guard-passing program | 1 | 1 | 1 | 0 | 0/0/0 | direct / direct | 100.00% | 1 |
| `9aaea919` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.44% | 0 |
| `9bbf930d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 97.52% | 0 |
| `a251c730` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `a25697e4` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.93% | 0 |
| `a32d8b75` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `a395ee82` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 87.60% | 0 |
| `a47bf94d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 93.49% | 0 |
| `a6f40cea` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `aa4ec2a5` | correct | Correct: guard-passing program | 1 | 1 | 0 | 1 | 1/0/0 | luna / direct | 100.00% | 1 |
| `abc82100` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 83.50% | 0 |
| `b0039139` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `b10624e5` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 84.23% | 0 |
| `b5ca7ac4` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 58.68% | 0 |
| `b6f77b65` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 80.21% | 0 |
| `b99e7126` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 97.50% | 0 |
| `b9e38dc0` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 68.07% | 0 |
| `bf45cf4b` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 0.00% | 0 |
| `c4d067a0` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 82.44% | 0 |
| `c7f57c3e` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 90.82% | 0 |
| `cb2d8a2c` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 85.15% | 0 |
| `cbebaa4b` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 87.06% | 0 |
| `d35bdbdc` | failed | Model timed out; unguarded fallback failed | 3 | 0 | 0 | 0 | 0/1/0 | direct / direct | 88.67% | 0 |
| `d59b0160` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | direct / direct | 73.83% | 0 |
| `d8e07eb2` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | direct / direct | 79.24% | 0 |
| `da515329` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 52.51% | 0 |
| `db0c5428` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 71.97% | 0 |
| `db695cfb` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 85.33% | 0 |
| `dbff022c` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 93.49% | 0 |
| `dd6b8c4b` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 87.60% | 0 |
| `de809cff` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/1 | fallback / fallback | 76.44% | 0 |
| `dfadab01` | failed | Synthesis returned but failed guards | 1 | 0 | 0 | 0 | 2/0/0 | luna / fallback | 92.50% | 0 |
| `e12f9a14` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 85.28% | 0 |
| `e3721c99` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 68.72% | 0 |
| `e376de54` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 98.83% | 0 |
| `e8686506` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |
| `e87109e9` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |
| `edb79dae` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |
| `eee78d87` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |
| `f560132c` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |
| `f931b4a8` | failed | Model timed out; unguarded fallback failed | 2 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 12.50% | 0 |
| `faa9f03d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 82.64% | 0 |
| `fc7cae8d` | failed | Model timed out; unguarded fallback failed | 1 | 0 | 0 | 0 | 0/1/0 | fallback / fallback | 0.00% | 0 |

## Method and limitations

The report joins the frozen submission and labelled public-evaluation files to read-only rows from `state.sqlite3`. Candidate verifier predictions were compared with labels after freeze to compute the candidate-oracle diagnostic. No candidate was rerun and neither frozen workspace nor submission was modified.

Primary diagnosis is intentionally operational. It cannot tell whether a timed-out model would eventually have returned a correct program, and exact public labels cannot be used to choose future competition outputs without creating evaluation leakage.

## Source integrity

- Evaluation report SHA-256: `35bcf555763f8cf764446429f05a83b4090ce89ce2ded5ad87558ccd65aca6fa`
- Frozen submission SHA-256: `539c9bdd7ed399f56979661c4e055553a1400cc918abc6270ec13b737b8ada7e`
- State database SHA-256 at analysis time: `cff946eeb4884451d9963bf145ae795861e70894ca7e929a297e36a1712f1d8d`
