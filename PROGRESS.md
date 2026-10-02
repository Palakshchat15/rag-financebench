# PROGRESS (resume file)

All commands run from this folder in PowerShell, using the project venv:
`.venv\Scripts\python ...`

## Status
- [x] 1. venv + pinned requirements.txt (Python 3.13.15). Ollama 0.34.4. qwen2.5:7b + qwen2.5:3b pulled.
- [x] 1b. Speeds: `.venv\Scripts\python scripts\bench_models.py` -> results/model_speed.json
      (first run, while downloads were running: 7B 16.5 prompt tok/s on 1.2k prompt, 6.2 gen tok/s; 3B 37 / 12).
      Re-run idle (final, in model_speed.json): 7B 65 tok/s prompt (40-tok prompt), 21 tok/s prompt on 1.2k prompt, 7-8 gen tok/s;
      3B 149 / 42 prompt tok/s, 13-17.5 gen tok/s.
- [x] 2. Data + MANIFEST + split: `.venv\Scripts\python scripts\download_data.py` (84 PDFs, 159 MB; dev 50 / test 100).
      All 189 gold evidence pages verified to align with our 0-indexed parse.
- [x] 3. Parse, chunk, index: `.venv\Scripts\python scripts\build_index.py` (log: results/build_index.log).
      12,013 pages. cs256 = 46,816 chunks. cs512 = 25,676, cs1024 = 15,047 chunks. Embedding took 4532 s / 4491 s / 2747 s.
      Resumable per size (skips sizes whose emb.npy exists).
- [x] 4. Router / retrieval / hybrid / reranker + pytest code written (src/routing.py, retrieval.py, scoring.py,
      validation.py; tests/ 82 passing). Router dev accuracy: company level 98%, company+year 96%.
- [x] 5. Dev grid DONE (90 configs, results/retrieval_dev.csv). Dev choice: cs1024, hybrid RRF weights (bm25 1, dense 2),
      NO rerank, routed company_year; dev page_hit@5 0.54, MRR 0.363. Written into config.yaml. Test scoring running
      Test DONE (results/retrieval_test.csv): chosen config test page_hit@5 0.55, @10 0.70, MRR 0.368; routing test 98%. Retrieval eval: `scripts\eval_retrieval.py --split dev` -> results/retrieval_choice.json, then copy the
      choice into config.yaml `retrieval:`, then `--split test`.
- [x] 6. Generation code written (src/generation.py, llm.py, judge.py, tracing.py). Not run yet.
- [x] 6b. Dev sanity check of prompt + judge on a few dev questions. DONE (prompt v4 + judge rubric v2 FROZEN).
      Finding 1: the v1 prompt made 7B refuse 3M FY2018 capex although "$1,577 million" (= gold) was in its
      context under "capital spending". Prompt v2 (rules 5/6 in src/generation.py): equivalent line-item
      names + "answer in the units asked". v1 dev output archived in results/gen_dev_prompt_v1/.
      num_thread benchmark (3B, 1.6k prompt): Ollama default 50 tok/s prefill beats 4/6/8/10/12 threads
      (38-45), so defaults kept. 7B RAG prompt (~5.2k tok) prefill ~15 tok/s -> ~6 min/question.
      Sanity run: `scripts\eval_generation.py --split dev --config oracle|rag --model qwen2.5:7b --ids <ids>`
      Prompt v2 on 4 dev oracle q (results/gen_dev_prompt_v2/): still over-refused (Activision: listed all
      inputs, then "I don't know"; AES: missed "cost of sales" = COGS). Cause: 'Answer:' first = decide before
      reasoning. Prompt v3: "Reasoning:" first, "Answer:" last; "CALCULATE if inputs are present"; synonyms
      incl. cost of sales. num_predict 512 -> 768. Scorer: uses the LAST 'Answer:' line; a reply that starts
      with 'Reasoning:' but has no Answer line (cut off) scores as no answer (+2 tests, 84 pass).
      v3 on 8 dev oracle q (results/gen_dev_prompt_v3/): 3M + Activision now correct; AES hallucinated an
      AES_2021 citation (validator flagged it); JPM refused the "is gross margin relevant" question; Ulta
      "none" answered as I don't know. Prompt v4: general financial knowledge allowed for relevance
      judgements (company facts still only from context); "section lists none -> answer none".
      Shared LLM lock (coordinator request): src/llm_lock.py creates ../.llm_lock (project, pid, start) around
      every eval_generation / score_generation (judge) run and around the whole run_eval_queue.py; waits if
      another project holds it.
      Judge sanity on the 10-q dev spot-check (results/score_dev_sanity.log): labels sensible except JnJ
      (cs256) judged partial for an extra, true net-earnings figure. Rubric v2 (src/judge.py): grade the final
      answer; don't penalise extra facts the gold doesn't mention. Re-judged: JnJ still partial -> stopped
      tuning (one example; judge validation on test will quantify). v1 judge cache in results/judge_dev_rubric_v1/.
- [x] 6c. Context budget (coordinator request: faster without losing quality; DEV only). DONE.
      `scripts\context_budget.py` -> results/context_budget_dev.csv (dev page_hit@k vs mean chunk tokens):
      current cs1024 top5 = 0.54 @ 3666 tok; cs512 top7 0.52 @ 3125; cs256 top9 0.56 @ 2080;
      cs256 top10 0.60 @ 2306 (-37% tokens). Spot-check (prompt v4, 7B, 10 dev q = every 5th dev q):
      both 3/10 correct (same 3), prompt 3418 vs 4694 tok, prefill 219 vs 297 s. DECISION: config.yaml
      retrieval chunk_size 256, top_k 10 (results/context_budget_choice.json). Test retrieval of this config
      (already in retrieval_test.csv ablation row): page_hit@10 0.66.
      NOTE: another project's OCR (doc_ai) ran CPU-heavy during part of the dev runs; coordinator had it
      stopped and it now takes .llm_lock for CPU-heavy batches. Latencies for the README must come from
      contention-free runs.
- [x] 7. DONE 2026-09-30 (all 6 test runs 100/100, scored). History: started 2026-09-29 03:15 (detached queue; check results/logs_queue.out and results/logs/*.log;
      if it died, just re-run the queue command, caches resume).
      DONE: closed_book 7b (100/100, scored: correct 3%, partial 3%, refused 76%). oracle 7b was at 74/100 when the
      previous session was cut off; queue re-started 2026-09-29 ~07:40 (resumes from caches).
      TODO for limitations: count answers that hit the 768-token cap (e.g. 04103).
      App fixes made meanwhile (verified by CDP screenshots): duplicate-label crash in the test ablation table,
      6-decimal formatting, cluttered/truncated charts -> Altair helper hbar(); context-budget section added;
      Deploy toolbar hidden. Screenshots: headless Edge --screenshot captures only Streamlit's skeleton, so
      screenshots are taken through the DevTools protocol (scratch script cdp_shot.py; wait 15 s per page). Generation evals (sequential, one at a time), queue runner (resumable, holds the LLM lock, scores after
      each job): `.venv\Scripts\python scripts\run_eval_queue.py` (order: closed_book 7b, oracle 7b, rag 3b,
      rag 7b, closed_book 3b, oracle 3b; logs in results/logs/). Single job:
      `scripts\eval_generation.py --split test --config {closed_book|oracle|rag} --model qwen2.5:7b|qwen2.5:3b`
      then `scripts\score_generation.py --split test` (runs the judge, cached in results/judge/).
- [x] 8. Judge validation DONE: `--make` (30 items, seed 7), hand labels in results/human_labels.jsonl
      (developer, blind to judge), `--score`. Judge alone 67% exact (kappa 0.49, results/
      judge_agreement_before_inline_answer_fix.json). Found scorer bug: inline '... Answer: I don't know'
      (3B) was not detected -> fixed final_answer_segment (src/scoring.py, +1 test, 85 pass), re-scored
      test with `scripts\score_generation.py --split test --no-judge` (cached judge labels, no LLM calls).
      Only 3B changed: rag 3B 14/2/42/42 -> 16/2/21/61, oracle 3B 24/1/23/52 -> 26/1/9/64 (C/P/I/R %).
      Final agreement 80% (kappa 0.70), binary 80%; remaining 6 disagreements all judge more lenient.
      Error analysis DONE: `scripts\error_analysis.py` (rag 7B -> results/error_analysis.json: retrieval
      miss 25, refusal 11, reading/calc 23, routing 0, judge disagreement 0; 59 not correct) and
      `--run test__rag__qwen2.5-3b.jsonl --out error_analysis_rag_3b.json` (32 / 37 / 15 of 84).
      768-token cap hits: rag 7B 1 (04103), oracle 7B 1, rag 3B 5, others 0.
- [x] 9. App verified 2026-09-30 (see end of file).
- [x] 10. README complete (all results tables, judge validation, error analysis, design decisions,
      limitations, next steps, 8 GB section).

**PROJECT COMPLETE (2026-09-30).** The scheduled task RAG_eval_queue no longer exists.

## Notes
- FinanceBench repo pinned to commit cc39aeb4 (config.yaml).
- Router rules checked on dev only. Aliases/tickers table is written for all 32 companies.

## Paused 2026-09-30 06:45 (owner request)
- Saved test answers: closed_book 7b 100/100 (scored), oracle 7b 100/100 (scored: 76% correct), rag 3b 100/100 (scored: 14% correct, 42% refused), rag 7b 40/100 (not scored yet).
- Resume: delete a stale ../.llm_lock if present, make sure Ollama answers on 127.0.0.1:11434, then start `.venv\Scripts\python scripts\run_eval_queue.py` as a separate process (Start-Process).  It resumes rag 7b at question 41, then closed_book 3b and oracle 3b.
- 2026-09-30 08:25: resumed via Windows Task Scheduler (task "RAG_eval_queue" -> scripts\run_queue_task.cmd, log results\logs_queue_task.out), because VS Code kills processes started from Claude when the extension restarts. To resume later: `schtasks /Run /TN "RAG_eval_queue"` (after clearing a stale ../.llm_lock). Delete the task when the evals are done: `schtasks /Delete /TN "RAG_eval_queue" /F`.

## App verification 2026-09-30
- `.venv\Scripts\streamlit run streamlit_app.py` (port 8601). Screenshots via headless Edge + CDP (scratch cdp_shot.py).
- Fixed: every Altair bar chart had collapsed bands/overlapping labels (Streamlit 1.64 shrinks a fixed pixel
  height) -> hbar() now uses alt.Step heights; deprecated use_container_width -> width="stretch"; judge text now
  shows judge-alone and final agreement; error-analysis section has a 7B/3B switch + token-cap count.
- Ask: one real 7B query (dev AMCOR adj. EBITDA): 266 s, 3409/116 tokens, citation valid, page viewer works;
  answer wrong (gave Adjusted EBIT $1,608m; gold EBITDA $2,018m).
- App RAM (excluding Ollama): 64 MB WS at start, 628 MB WS / 1.1 GB private after the query. Server stopped.
- pytest: 85 passed.
