"""FinanceBench RAG assistant: Ask / Evaluation / Monitoring.

Run:  .venv\\Scripts\\streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import load_config, path, questions_for  # noqa: E402

st.set_page_config(page_title="FinanceBench RAG", page_icon=":bar_chart:", layout="wide")

ATTRIBUTION = ("Data: **FinanceBench** open-source sample by Patronus AI (Islam et al., 2023), "
               "CC BY-NC 4.0. Non-commercial portfolio project.")
CFG = load_config()
RES = path("results")


# ---------------------------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading index, embedder and reranker ...")
def get_pipeline():
    from src.generation import Pipeline
    from src.indexing import embed_query
    from src.retrieval import get_reranker

    p = Pipeline(trace_source="app")
    embed_query("warm up")          # load the embedding model once
    if p.rc["rerank"]:
        get_reranker()
    return p


@st.cache_data
def load_csv(name: str) -> pd.DataFrame | None:
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_json(name: str):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_jsonl(rel: str) -> list[dict]:
    p = RES / rel
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


@st.cache_data(max_entries=64)
def page_png(doc_name: str, page: int) -> bytes:
    from src.ingest import render_page_png

    return render_page_png(path("pdfs") / f"{doc_name}.pdf", page)


_CIT = re.compile(r"\[([^\[\]]*\bp\.?\s*\d+[^\[\]]*)\]")


def highlight_citations(text: str) -> str:
    return _CIT.sub(lambda m: f"**`[{m.group(1)}]`**", text)


# ---------------------------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------------------------
def page_ask():
    from src.llm import OllamaUnavailable, check_ollama

    st.title("Ask the filings")
    st.caption("Questions about 84 SEC filings (10-K, 10-Q, 8-K, earnings releases) of 32 US "
               "companies. Answers use only retrieved pages and cite them as [DOC p.N] "
               "(N = 0-indexed PDF page, as in FinanceBench).")

    with st.sidebar:
        st.subheader("Settings")
        models = CFG["llm"]["models_evaluated"]
        default = models.index(CFG["llm"]["model"]) if CFG["llm"]["model"] in models else 0
        model = st.selectbox("Model (Ollama)", models, index=default)
        mode = st.radio("Retrieval mode", ["routed", "shared"],
                        index=0 if CFG["retrieval"]["mode"] == "routed" else 1,
                        help="routed: search only the filings matched by company/year. "
                             "shared: search all 84 filings.")
        rc = CFG["retrieval"]
        st.caption(f"Retrieval: {rc['method']}{' + rerank' if rc['rerank'] else ''}, "
                   f"{rc['chunk_size']}-token chunks, top {rc['top_k']} (chosen on dev).")
        try:
            check_ollama(model)
            st.success(f"Ollama OK: {model}")
            ollama_ok = True
        except OllamaUnavailable as e:
            st.error(str(e))
            ollama_ok = False

    examples = [q["question"] for q in questions_for("dev")][:: 5]
    ex = st.selectbox("Example questions (from the dev split)", ["(type your own)"] + examples)
    q = st.text_area("Question", value="" if ex == "(type your own)" else ex, height=90)
    go = st.button("Ask", type="primary", disabled=not ollama_ok or not q.strip())

    if go:
        pipe = get_pipeline()
        with st.spinner(f"Retrieving and generating with {model} (CPU inference can take a minute or two) ..."):
            try:
                ans = pipe.answer(q.strip(), model=model, config="rag", mode=mode)
                st.session_state["last"] = ans
                st.session_state.pop("cite_sel", None)
            except OllamaUnavailable as e:
                st.error(str(e))
                return
    ans = st.session_state.get("last")
    if not ans:
        st.info("Pick an example or type a question, then press Ask.")
        return

    st.subheader("Answer")
    st.markdown(highlight_citations(ans.answer))
    if ans.invalid_citations:
        st.warning("Citation check: these citations are NOT among the retrieved pages "
                   f"(possible hallucination): {', '.join(f'{d} p.{p}' for d, p in ans.invalid_citations)}")
    elif ans.citations:
        st.success(f"Citation check: all {len(ans.citations)} citation(s) point to retrieved pages.")
    else:
        st.info("No citations in the answer.")

    t = ans.timings
    cols = st.columns(6)
    for c, (lab, key) in zip(cols, [("Route", "route_s"), ("Retrieve", "retrieve_s"), ("Rerank", "rerank_s"),
                                    ("Generate", "generate_s"), ("Total", "total_s")]):
        c.metric(lab, f"{t.get(key, 0):.2f} s")
    cols[5].metric("Tokens in / out", f"{ans.prompt_tokens} / {ans.completion_tokens}")
    if ans.route_candidates:
        st.caption("Routed to: " + ", ".join(ans.route_candidates))
    elif mode == "routed":
        st.caption("Router found no company match: searched all filings.")

    if ans.citations:
        st.markdown("**Cited pages** (click to view the page)")
        labels = [f"{d} p.{p}" for d, p in ans.citations]
        sel = st.pills("cited pages", labels, key="cite_sel", label_visibility="collapsed")
        if sel:
            d, p = sel.rsplit(" p.", 1)
            show_page(d, int(p))

    with st.expander(f"Retrieved chunks ({len(ans.blocks)}) given to the model", expanded=False):
        for i, b in enumerate(ans.blocks, 1):
            st.markdown(f"**{i}. {b.doc_name} p.{b.page}** (score {b.score:.3f}, `{b.chunk_id}`)")
            st.text(b.text[:3000])


def show_page(doc: str, page: int):
    pdf = path("pdfs") / f"{doc}.pdf"
    if not pdf.exists():
        st.warning(f"{doc} is not in the corpus (the model cited a document that does not exist).")
        return
    c1, c2 = st.columns([3, 2])
    try:
        c1.image(page_png(doc, page), caption=f"{doc}, PDF page index {page}")
    except Exception as e:  # noqa: BLE001
        c1.warning(f"Could not render page: {e}")
    c2.text_area("Page text", get_pipeline().page_text(doc, page), height=600)


# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------
# Categorical slots (dark mode) in fixed order: blue, orange, aqua.
SERIES = ["#3987e5", "#d95926", "#199e70"]


def hbar(df: pd.DataFrame, label: str, metrics: list[str], title: str = "", pct: bool = True,
         label_width: int = 420):
    """Grouped horizontal bars with full category labels, hover tooltips, recessive grid.
    Height is set per category with alt.Step: Streamlit 1.64 shrinks a fixed pixel height,
    which collapsed the bands (checked with screenshots)."""
    import altair as alt

    long = df[[label] + metrics].melt(id_vars=label, var_name="metric", value_name="value")
    order = list(df[label])
    n = len(metrics)
    enc_color = (alt.Color("metric:N", scale=alt.Scale(domain=metrics, range=SERIES[:n]),
                           legend=alt.Legend(orient="top", title=None)) if n > 1 else alt.value(SERIES[0]))
    ch = (alt.Chart(long, title=title)
          .mark_bar(cornerRadiusEnd=3)
          .encode(y=alt.Y(f"{label}:N", sort=order, title=None,
                          axis=alt.Axis(labelLimit=label_width, labelFontSize=12)),
                  x=alt.X("value:Q", title="%" if pct else None, axis=alt.Axis(grid=True, gridOpacity=0.15)),
                  color=enc_color,
                  tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("metric:N"),
                           alt.Tooltip("value:Q", format=".1f")])
          .properties(height=alt.Step(30 if n == 1 else 14)))
    if n > 1:
        ch = ch.encode(yOffset=alt.YOffset("metric:N", sort=metrics))
    st.altair_chart(ch.configure_view(strokeWidth=0), width="stretch")


def fmt_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    for c in cols:
        if c in d:
            d[c] = (d[c] * 100).round(1)
    return d


def page_eval():
    st.title("Evaluation (saved results; nothing is re-run here)")
    st.caption("Split: 150 FinanceBench questions -> dev 50 / test 100, stratified by question type "
               "(seed 42). Every tuning choice was made on dev; test numbers were scored once.")

    choice = load_json("retrieval_choice.json")
    routing = load_json("routing.json")
    dev, test = load_csv("retrieval_dev.csv"), load_csv("retrieval_test.csv")
    st.header("1. Retrieval")
    if choice:
        c = choice["chosen"]
        st.markdown(f"**Dev-chosen config:** {c['method']}"
                    f"{' + cross-encoder rerank (depth ' + str(c['rerank_depth']) + ')' if c['rerank'] else ''}, "
                    f"{c['mode']} mode, {c['size']}-token chunks, RRF k={c['rrf_k']}, "
                    f"weights (bm25, dense)={tuple(c['weights'])}. Criterion: {choice['criterion']}. "
                    f"The generation runs then use the same method with a smaller context: "
                    f"**{CFG['retrieval']['chunk_size']}-token chunks, top {CFG['retrieval']['top_k']}** "
                    "(context-budget decision on dev, below).")
    if routing:
        rr = pd.DataFrame([{"split": sp, "level": lv, **m} for sp, d in routing.items()
                           for lv, m in d.items()]).set_index(["split", "level"])
        st.markdown("**Router accuracy** (gold filing among the routed candidates)")
        st.dataframe(fmt_pct(rr, [c for c in rr.columns if c != "n" and c != "mean_candidates"]),
                     width="stretch")
    metric_cols = ["page_hit@1", "page_hit@3", "page_hit@5", "page_hit@10", "mrr@10",
                   "doc_hit@1", "doc_hit@5"]
    if test is not None:
        st.markdown("**Test set (100 q): ablation rows at cs1024 plus the chosen method at cs256/cs512** "
                    "(evidence-page hit@k, %)")
        t = test.copy()
        used = CFG["retrieval"]
        is_used = ((t["size"] == used["chunk_size"]) & (t.method == used["method"]) & (t.rerank == used["rerank"])
                   & (t["mode"] == f"routed:{used['route_level']}") & (t.weights == str([float(w) for w in used["weights"]])))
        t["config"] = (t.method + t.rerank.map({True: "+rerank", False: ""}) + " / " + t["mode"] + " / cs"
                       + t["size"].astype(str)
                       + t.weights.map(lambda w: "" if w == "[1.0, 1.0]" else f" / w{w}")
                       + t.chosen.map({True: "  [page_hit@5 pick]", False: ""})
                       + is_used.map({True: f"  [USED for RAG, top {used['top_k']}]", False: ""}))
        show = fmt_pct(t.set_index("config")[metric_cols + ["sec_per_query"]], metric_cols)
        st.dataframe(show.style.highlight_max(subset=metric_cols, color="#1f6f43")
                     .format("{:.1f}", subset=metric_cols).format("{:.3f}", subset=["sec_per_query"]),
                     width="stretch")
        hbar(show.reset_index(), "config", ["page_hit@5", "page_hit@10"],
             "Test evidence-page hit rate (%) by retrieval config", label_width=460)
    budget = load_csv("context_budget_dev.csv")
    bchoice = load_json("context_budget_choice.json")
    if budget is not None:
        st.markdown("**Context budget (dev):** how much evidence recall each prompt size buys. "
                    "The page_hit@5 pick above (cs1024, top 5) needs ~3.7k context tokens; "
                    "top 10 of the 256-token index reaches higher dev recall with ~37% fewer tokens, "
                    "so the RAG answers use **cs256, top 10**"
                    + (f" (test page_hit@10 of this config: 66%)." if bchoice else "."))
        piv = budget.pivot(index="k", columns="chunk_size", values="page_hit")
        tok = budget.pivot(index="k", columns="chunk_size", values="ctx_tokens_mean")
        c1, c2 = st.columns(2)
        c1.caption("Dev evidence-page hit rate by k (columns: chunk size)")
        c1.line_chart(piv.rename(columns=lambda c: f"cs{c}"), height=260, color=SERIES,
                      x_label="k (chunks in the prompt)", y_label="page hit rate")
        c2.caption("Mean context tokens put in the prompt by k")
        c2.line_chart(tok.rename(columns=lambda c: f"cs{c}"), height=260, color=SERIES,
                      x_label="k (chunks in the prompt)", y_label="tokens")
        if bchoice:
            with st.expander("Context-budget decision details (dev spot-check)"):
                st.json(bchoice)
    if dev is not None:
        with st.expander("Full dev ablation grid (all configs; used for the choice)"):
            d = dev.copy()
            d["config"] = (d.method + d.rerank.map({True: "+rerank", False: ""}) + " / " + d["mode"] + " / cs"
                           + d["size"].astype(str) + " / w" + d.weights.astype(str) + " / rd" + d.rerank_depth.astype(str))
            st.dataframe(fmt_pct(d.set_index("config")[metric_cols + ["sec_per_query"]], metric_cols)
                         .sort_values("page_hit@5", ascending=False), width="stretch")

    st.header("2. Generation (test, 100 questions)")
    gen = load_csv("generation_test.csv")
    if gen is None:
        st.info("No generation results saved yet.")
    else:
        cols = ["config", "model", "n", "correct", "partially_correct", "incorrect", "refusal_rate",
                "correct_metrics-generated", "correct_domain-relevant", "correct_novel-generated",
                "citation_validity", "cites_gold_page", "gold_page_in_context", "p50_total_s", "p95_total_s"]
        g = gen[[c for c in cols if c in gen]].copy()
        pc = [c for c in g.columns if c not in ("config", "model", "n", "p50_total_s", "p95_total_s")]
        g = fmt_pct(g, pc)
        order = {"closed_book": 0, "rag": 1, "oracle": 2}
        g = g.sort_values(["config", "model"], key=lambda s: s.map(order) if s.name == "config" else s)
        st.dataframe(g.set_index(["config", "model"]), width="stretch")
        hbar(g.assign(run=g.config + " / " + g.model), "run", ["correct", "partially_correct"],
             "Share of test questions (%) by config and model", label_width=240)
        st.caption("Correct = strict (numeric within 1% or judge 'correct'). Closed-book = no context; "
                   "oracle = gold evidence page(s) given; rag = dev-chosen retrieval, top-k chunks.")
    ja = load_json("judge_agreement.json")
    ja0 = load_json("judge_agreement_before_inline_answer_fix.json")
    if ja:
        txt = (f"**Judge vs hand labels** ({ja['n']} judge-scored test answers, labelled blind by the developer, "
               "not independent annotators): ")
        if ja0:
            txt += (f"judge alone {ja0['agreement']*100:.0f}% exact agreement (kappa {ja0['kappa']:.2f}); "
                    f"final labels after the inline-'Answer:' scorer fix {ja['agreement']*100:.0f}% (kappa {ja['kappa']:.2f}); ")
        else:
            txt += f"{ja['agreement']*100:.0f}% exact agreement (kappa {ja['kappa']:.2f}); "
        txt += (f"correct vs not-correct {ja['binary_agreement']*100:.0f}% (kappa {ja['binary_kappa']:.2f}). "
                "Every remaining disagreement is the judge being more lenient (it said 'correct' where the "
                "hand label is partial or incorrect), so the strict 'correct' rate of judge-scored answers is optimistic.")
        st.markdown(txt)

    st.header("3. Error analysis (test, full RAG)")
    ea_runs = {"qwen2.5:7b": "error_analysis.json", "qwen2.5:3b": "error_analysis_rag_3b.json"}
    ea_model = st.radio("Model", [m for m, f in ea_runs.items() if (RES / f).exists()], horizontal=True,
                        key="ea_model")
    ea = load_json(ea_runs[ea_model]) if ea_model else None
    if ea:
        cap = ea.get("hit_token_cap")
        if cap:
            st.caption(f"{cap['count']} of {ea['n_questions']} answers hit the {cap['cap']}-token generation cap.")
        st.caption(ea.get("run", ""))
        b = pd.DataFrame([{"bucket": k, "count": v["count"]} for k, v in ea["buckets"].items()]).set_index("bucket")
        c1, c2 = st.columns([1, 2])
        c1.dataframe(b, width="stretch")
        with c2:
            hbar(b.reset_index(), "bucket", ["count"], "Not-correct answers by cause (count)", pct=False,
                 label_width=220)
        for k, v in ea["buckets"].items():
            if v["examples"]:
                with st.expander(f"{k}: {v['count']} (examples)"):
                    for e in v["examples"]:
                        st.markdown(f"**Q:** {e['question']}\n\n**Gold:** {e['gold']}\n\n**Model:** {e['answer'][:800]}\n\n"
                                    f"*{e['note']}*")
                        st.divider()

    st.header("4. Per-question browser (test)")
    runs = sorted(p.name for p in (RES / "scored").glob("test__*.jsonl")) if (RES / "scored").exists() else []
    if runs:
        run = st.selectbox("Run", runs, index=next((i for i, r in enumerate(runs) if "__rag__" in r and "7b" in r), 0))
        scored = {r["id"]: r for r in load_jsonl(f"scored/{run}")}
        gen_rows = {r["id"]: r for r in load_jsonl(f"gen/{run}")}
        qs = {q["financebench_id"]: q for q in questions_for("test")}
        ids = sorted(scored)
        filt = st.radio("Show", ["all", "correct", "partially_correct", "incorrect", "refused"], horizontal=True)
        ids = [i for i in ids if filt == "all" or scored[i]["label"] == filt]
        if ids:
            qid = st.selectbox(f"Question ({len(ids)})", ids, format_func=lambda i: f"{i}: {qs[i]['question'][:90]}")
            q, s, g = qs[qid], scored[qid], gen_rows[qid]
            st.markdown(f"**Question** ({q['question_type']}): {q['question']}")
            c1, c2 = st.columns(2)
            c1.markdown(f"**Gold answer:** {q['answer']}")
            c1.markdown("**Evidence page(s):** " + ", ".join(f"{e['doc_name']} p.{e['evidence_page_num']}" for e in q["evidence"]))
            c2.markdown(f"**Label:** `{s['label']}` via {s['method']}" + (f" ({s['reason']})" if s["reason"] else ""))
            c2.markdown("**Retrieved / given pages:** " + (", ".join(f"{c[1]} p.{c[2]}" for c in g["context"]) or "none"))
            st.markdown("**Model answer:**")
            st.markdown(highlight_citations(g["answer"]))


# ---------------------------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------------------------
def page_monitor():
    from src.tracing import Tracer

    st.title("Monitoring (data/traces.db)")
    db = path("traces_db")
    if not db.exists():
        st.info("No traces yet.")
        return
    df = Tracer(db).read()
    if df.empty:
        st.info("No traces yet.")
        return
    src = st.radio("Source", ["all", "app", "eval"], horizontal=True)
    if src != "all":
        df = df[df.source == src]
    df["time"] = pd.to_datetime(df.ts, unit="s")
    c = st.columns(4)
    c[0].metric("Queries", len(df))
    c[1].metric("p50 total latency", f"{df.t_total.median():.1f} s" if len(df) else "-")
    c[2].metric("p95 total latency", f"{df.t_total.quantile(0.95):.1f} s" if len(df) else "-")
    c[3].metric("Answers with all citations valid", f"{df.citations_valid.mean()*100:.0f}%" if len(df) else "-")

    st.subheader("Latency by stage (seconds)")
    stages = ["t_route", "t_retrieve", "t_rerank", "t_generate", "t_total"]
    grp = df.groupby(["config", "model"])[stages]
    lat = pd.concat({"p50": grp.median(), "p95": grp.quantile(0.95)}, axis=1).round(2)
    st.dataframe(lat, width="stretch")
    rag = df[df.config == "rag"]
    if len(rag):
        med = rag[["t_route", "t_retrieve", "t_rerank"]].median().rename("p50_s").reset_index()
        hbar(med.rename(columns={"index": "stage"}), "stage", ["p50_s"],
             "Retrieval-side stages for RAG queries (p50, seconds)", pct=False, label_width=120)

    st.subheader("Tokens per query")
    tok = df.groupby(["config", "model"])[["prompt_tokens", "completion_tokens"]].mean().round(0)
    st.dataframe(tok, width="stretch")
    hbar(tok.reset_index().assign(run=lambda d: d.config + " / " + d.model), "run",
         ["prompt_tokens", "completion_tokens"], "Mean tokens per query", pct=False, label_width=240)

    st.subheader("Recent queries")
    recent = df.head(25)[["time", "source", "config", "mode", "model", "question", "t_total",
                          "prompt_tokens", "completion_tokens", "citations_valid", "invalid_citations"]]
    st.dataframe(recent, width="stretch", hide_index=True)


nav = st.navigation([
    st.Page(page_ask, title="Ask", default=True),          # served at the root URL (url_path="ask" gave "Page not found")
    st.Page(page_eval, title="Evaluation", url_path="evaluation"),
    st.Page(page_monitor, title="Monitoring", url_path="monitoring"),
])
with st.sidebar:
    st.caption(ATTRIBUTION)
nav.run()
