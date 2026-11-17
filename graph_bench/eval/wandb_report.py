"""Build (or rebuild in place) the "Dashboard Summary" W&B report of the Mimir graph benchmark, project Mimir-Graph-RLVR.

One report, sections in reading order: test overall, test by file (trained tasks, then harder / held-out files),
whether the model finished its answer, anonymised entities, val pass@1 vs pass@16 (headroom for RL), GRPO training
curves, the baselines after the same SFT, OSM. Every panel grid has its own run set (W&B filters on the run config
written by log_wandb.py: arm, model_name, condition), so test and val runs are never mixed in one chart, and runs tagged
`superseded` are left out. New evaluation runs (E1p, E2c, E2p, E3c, ...) appear by themselves; rerun this script only to
change the layout or the colours (one colour per model, read from the runs that exist when it runs).

Needs the wandb-workspaces package, which the repo venvs do not have:
    uv venv /tmp/wsvenv --python .venv/bin/python && VIRTUAL_ENV=/tmp/wsvenv uv pip install wandb-workspaces
    /tmp/wsvenv/bin/python graph_bench/eval/wandb_report.py            # creates the report, prints its URL
    /tmp/wsvenv/bin/python graph_bench/eval/wandb_report.py --url URL  # rebuilds that report in place
The URL of the last build is kept in <root>/wandb_report_url.txt and used by default.
"""
import argparse
from pathlib import Path

import wandb
import wandb_workspaces.reports.v2 as wr

ENTITY, PROJECT = "m-ortega-p-le-l-onard-de-vinci", "Mimir-Graph-RLVR"
ROOT = Path("/work/dfm/marius-ortega/dfm-hrm-graphs/mimir_bench")
TITLE = "Dashboard Summary"
NOT_SUPERSEDED = "Tags() not in ['superseded']"

# One colour per model (arm), the same in every section. Old arms in greys / muted, the relaunch in strong colours.
COLORS = {"e0": "#bdbdbd", "e1": "#1f77b4", "e1p": "#6baed6", "e2": "#ff7f0e", "e2c": "#d62728", "e2p": "#e377c2",
          "e3": "#2ca02c", "e3b": "#98df8a", "e3c": "#006d2c", "e3e1": "#17becf", "e4": "#9467bd", "e4sft": "#c5b0d5", "e5": "#8c564b",
          "e5sft": "#c49c94"}
BASELINE_COLORS = {"mimir15": "#1f77b4", "qwen25_7b": "#ff7f0e", "olmo3_7b": "#2ca02c", "qwen35_4b": "#9467bd"}
GRPO_COLORS = {"e3_grpo_e2": "#2ca02c", "e3c_grpo_e2c": "#006d2c", "e3c_grpo_e2p": "#006d2c",
               "e3_grpo_e1": "#17becf", "e3b_grpo_base": "#98df8a", "e4_grpo_clutrr": "#9467bd",
               "e5_grpo_multi": "#8c564b"}


def runset(name: str, filters: str) -> wr.Runset:
    return wr.Runset(entity=ENTITY, project=PROJECT, name=name, filters=filters,
                     order=[wr.OrderBy(name=wr.Config("arm"), ascending=True)])


def bars(title: str, metrics: list[str], legend: str = "${config:arm}") -> wr.BarPlot:
    return wr.BarPlot(title=title, metrics=metrics, orientation="h", range_x=(0, 1), legend_template=legend,
                      max_runs_to_show=40, max_bars_to_show=80, font_size="small")


def lines(title: str, y: list[str]) -> wr.LinePlot:
    return wr.LinePlot(title=title, x="Step", y=y, legend_template="${run:displayName}", font_size="small",
                       max_runs_to_show=10)


def grid(rs: wr.Runset, panels: list, colors: dict, per_row: int = 3, h: int = 8) -> wr.PanelGrid:
    w = 24 // per_row
    for i, p in enumerate(panels):
        p.layout = wr.Layout(x=(i % per_row) * w, y=(i // per_row) * h, w=w, h=h)
    return wr.PanelGrid(runsets=[rs], panels=panels, custom_run_colors=colors)


def colors_by(runs, key) -> dict:
    """Run id -> colour, from a function of the run (its arm, model or name)."""
    out = {}
    for r in runs:
        c = key(r)
        if c:
            out[r.id] = c
    return out


def build() -> list:
    api = wandb.Api(timeout=60)
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", per_page=300))
    by_arm = colors_by(runs, lambda r: COLORS.get(r.config.get("arm")))
    by_model = colors_by(runs, lambda r: BASELINE_COLORS.get(r.config.get("model_name")))
    by_name = colors_by(runs, lambda r: GRPO_COLORS.get(r.name))

    mimir = f"Config('model_name') = 'mimir15' and {NOT_SUPERSEDED}"
    test = runset("Mimir v1.5, test (greedy)", f"{mimir} and Config('condition') in ['named', 'named_t3k']")
    anon = runset("Mimir v1.5, test, entities anonymised", f"{mimir} and Config('condition') in ['anon', 'anon_t3k']")
    val = runset("Mimir v1.5, val (16 samples)", f"{mimir} and Config('condition') in ['val_pass16', 'val_pass16_t2k']")
    hold = runset("Mimir v1.5, never-scored test rows",
                  f"{mimir} and Config('condition') in ['holdback', 'holdback_t3k']")
    osm = runset("Mimir v1.5, OSM test", f"{mimir} and Config('condition') = 'osm_t3k'")
    base = runset("All models after the same SFT, test",
                  f"Config('model_name') in {list(BASELINE_COLORS)} and Config('arm') in ['e0', 'e1', 'e2'] "
                  f"and Config('condition') in ['named', 'named_t3k'] and {NOT_SUPERSEDED}")
    grpo = wr.Runset(entity=ENTITY, project=PROJECT, name="GRPO training runs",
                     filters="Name in ['e3_grpo_e2', 'e3b_grpo_base', 'e4_grpo_clutrr', 'e5_grpo_multi', "
                             "'e3c_grpo_e2c', 'e3c_grpo_e2p', 'e3_grpo_e1']")

    acc = lambda f: f"split/{f}/acc"  # noqa: E731
    p = wr.P
    return [
        wr.TableOfContents(),
        p("How to read: one bar = one model, one colour per model in every chart (e0 base model, e1 answer-only SFT, "
          "e1p e1 + 523 WebQSP questions without a path, e2 trace SFT, e2c e2 with clean CLRS traces, e2p e2c + the 523 "
          "questions, e3 GRPO from e2, e3b GRPO from the base, e3c GRPO from the better of e2c / e2p, e3e1 GRPO from e1, "
          "e4 / e5 CLUTRR / "
          "new-dataset adaptations). Accuracy from 0 to 1, one seed: differences under 0.02 are noise. Test charts and "
          "val charts never share a panel. e5sft trained on CWQ questions built from WebQSP test questions: its "
          "WebQSP test score is not trustworthy. The breakdown by question type (gold path in the facts or not) is not on "
          "W&B: mimir_bench/trace_gap_v2*.txt and mimir_bench/rescore_20261116/."),

        wr.H1("1. Test: overall"),
        p("Macro accuracy = mean over the 7 datasets of the test set (the 'Macro named' of the wiki). "
          "e0 / e1 / e1p answer with up to 1,024 tokens, trace models with up to 3,072."),
        grid(test, [bars("Macro accuracy, test (7 datasets)", ["macro_acc"])], by_arm, per_row=1, h=9),

        wr.H1("2. Test by file: tasks seen in training"),
        p("GraphQA test_fresh is used instead of GraphQA test, whose graphs also occur in training."),
        grid(test, [bars("GraphQA (fresh graphs)", [acc("graphqa/test_fresh")]),
                    bars("CLRS (graph sizes of training)", [acc("clrs/test")]),
                    bars("MetaQA (gold subgraph, 1-3 hops)", [acc("metaqa/test")]),
                    bars("KQA Pro (gold subgraph)", [acc("kqapro/test")]),
                    bars("WebQSP", [acc("webqsp/test")])], by_arm),

        wr.H1("3. Test by file: harder or held out"),
        p("Larger graphs, retrieved (noisier) subgraphs, and datasets never trained on (CWQ, CLUTRR)."),
        grid(test, [bars("GraphQA, larger graphs", [acc("graphqa/test_large")]),
                    bars("CLRS, larger graphs (13-20 nodes)", [acc("clrs/test_ood")]),
                    bars("MetaQA, retrieved subgraph", [acc("metaqa/test_retrieved")]),
                    bars("KQA Pro, retrieved subgraph", [acc("kqapro/test_retrieved")]),
                    bars("CWQ (never trained on)", [acc("cwq/test")]),
                    bars("CLUTRR (never trained on)", [acc("clutrr/test")])], by_arm),

        wr.H1("4. Did the model finish its answer?"),
        p("Share of answers that contain an 'Answer:' line. Below 1 = the model looped or ran out of tokens; those "
          "rows count as wrong."),
        grid(test, [bars("Finished answers: CLRS, larger graphs", ["split/clrs/test_ood/answer_rate"]),
                    bars("Finished answers: WebQSP", ["split/webqsp/test/answer_rate"]),
                    bars("Finished answers: GraphQA, larger graphs", ["split/graphqa/test_large/answer_rate"])],
             by_arm),

        wr.H1("5. Test with entity names replaced"),
        p("Same KGQA questions with every entity renamed: a model that relied on memorised names drops here."),
        grid(anon, [bars("Macro accuracy, anonymised (3 datasets)", ["macro_acc"]),
                    bars("MetaQA, anonymised", [acc("metaqa/test")]),
                    bars("KQA Pro, anonymised", [acc("kqapro/test")]),
                    bars("CWQ, anonymised", [acc("cwq/test")])], by_arm, per_row=2),

        wr.H1("6. Never-scored test questions"),
        p("Test questions left out of every earlier evaluation (CWQ 2,531, WebQSP 628, CLRS 200, GraphQA fresh / "
          "large 100 each). The relaunch's fixes were designed without them: a gain seen in sections 1-3 should also "
          "show here. Empty until the relaunch queue evaluates them."),
        grid(hold, [bars("CWQ", [acc("cwq/test")]), bars("WebQSP", [acc("webqsp/test")]),
                    bars("CLRS", [acc("clrs/test")]), bars("GraphQA, fresh graphs", [acc("graphqa/test_fresh")]),
                    bars("GraphQA, larger graphs", [acc("graphqa/test_large")])], by_arm),

        wr.H1("7. Val: accuracy and headroom for RL"),
        p("16 sampled answers per val question. pass@1 = chance that one answer is right; pass@16 = chance that at "
          "least one of 16 is right. GRPO mostly raises pass@1 towards pass@16, so the gap between the two bars of a "
          "model is roughly what RL can still add. Not comparable with the test charts (sampled, not greedy)."),
        grid(val, [bars(f"{name}: pass@1 vs pass@16",
                        [f"split/{ds}/val/pass@1", f"split/{ds}/val/pass@16"], legend="${config:arm}")
                   for ds, name in (("graphqa", "GraphQA"), ("clrs", "CLRS"), ("metaqa", "MetaQA"),
                                    ("kqapro", "KQA Pro"), ("webqsp", "WebQSP"))], by_arm),

        wr.H1("8. GRPO training"),
        p("Each curve is one GRPO run over its training steps. Val accuracy is measured every 25 steps on about 60 "
          "val questions per dataset (greedy). Training reward = share of rollouts the reward judged right."),
        grid(grpo, [lines(f"Val accuracy during GRPO: {name}", [f"val-core/{ds}/acc/mean@1"])
                    for ds, name in (("graphqa", "GraphQA"), ("clrs", "CLRS"), ("metaqa", "MetaQA"),
                                     ("kqapro", "KQA Pro"), ("webqsp", "WebQSP"), ("clutrr", "CLUTRR"))]
             + [lines("Training reward (mean over rollouts)", ["critic/score/mean"]),
                lines("Answer length (tokens)", ["response_length/mean"]),
                lines("Answers cut at the length limit (share)", ["response_length/clip_ratio"])], by_name),

        wr.H1("9. Mimir against the baselines"),
        p("Same data and recipe for every model: e0 zero-shot, e1 answer-only SFT, e2 trace SFT. Colour = model."),
        grid(base, [bars("Macro accuracy, test", ["macro_acc"], legend="${config:arm} ${config:model_name}")],
             by_model, per_row=1, h=11),

        wr.H1("10. OSM spatial questions (never trained on, except e5)"),
        grid(osm, [bars("OSM test accuracy", [acc("osm/test")])], by_arm, per_row=2),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=None, help="Report to rebuild in place (default: the last build's URL).")
    ap.add_argument("--new", action="store_true", help="Create a new report even if one was built before.")
    args = ap.parse_args()
    url_file = ROOT / "wandb_report_url.txt"
    url = args.url or (None if args.new or not url_file.exists() else url_file.read_text().strip())
    blocks = build()
    description = ("Mimir v1.5 graph benchmark: test, val and GRPO training at a glance. Built by "
                   "graph_bench/eval/wandb_report.py.")
    if url:
        report = wr.Report.from_url(url)
        report.title, report.description, report.blocks, report.width = TITLE, description, blocks, "fluid"
    else:
        report = wr.Report(entity=ENTITY, project=PROJECT, title=TITLE, description=description, blocks=blocks,
                           width="fluid")
    report.save()
    url_file.write_text(report.url + "\n")
    print(report.url)


if __name__ == "__main__":
    main()
