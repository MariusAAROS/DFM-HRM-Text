---
name: wiki-log
description: Document work in this repo's OKF wiki, log-book style, under wiki/pages/hrm-graph/ - the implementation choices and why, what failed (symptom, cause, fix), what worked (verified numbers), in dated sections. First finds whether an existing page covers the topic and appends there; creates a new page only if none does. Use whenever the user asks to document, write up, log, record or "put in the wiki" an experiment, a fix, a result or a design decision, or at the end of a piece of work whose findings should survive the session, even if they don't say "wiki".
---

# Wiki log (hrm-graph)

The wiki (`wiki/`) is an OKF bundle; its rules are in `wiki/schema.md`. In this project, pages go in
`wiki/pages/hrm-graph/`. A page is a log book for one topic: a short header saying what the topic is and where
its code/data/W&B live, then dated sections appended in chronological order. Readers come back to it weeks
later to learn what was decided, what broke and what the numbers were, so write for that reader.

## 1. Find the page

Read `wiki/pages/hrm-graph/index.md` (one line per page) and list the headings of likely candidates
(`grep -n '^#' wiki/pages/hrm-graph/<page>.md`). Read only the sections you need.

- Same experiment, package, dataset pipeline or W&B group as an existing page → append to that page.
- A new, independently useful topic (its own code, question or W&B group) → new page. Don't stretch a page
  to cover something unrelated: the schema wants one retrievable purpose per page.
- A page over ~40 KB (`wc -c`) is near the 50 KB limit: put the new topic on a new page and link it from the
  old one.

## 2. Write the entry

Match the existing pages (e.g. `graph-structure-injection.md`): short sentences, bullets, bold lead-ins,
tables for results, code paths in backticks.

**Appending**: add a section at the end (or under the section it extends), headed
`## <What happened> (YYYY-MM-DD)`. Use the date of the work, absolute.

**New page**: copy this skeleton.

```markdown
---
type: Runbook
title: <Title>
description: <One sentence: what this topic is and what the page records.>
tags: [graphs, <dataset>, <model>, ...]
status: stable
last_updated: YYYY-MM-DD
confidence: high
---
# <Title>

<The question or purpose, 2-3 sentences.>

- Code: `<path>`.
- Tests / queue / data: `<paths>`.
- W&B: project `<P>`, group `<G>`.

## Design (YYYY-MM-DD)
```

What each section should carry, as it applies:
- **Choices**: what was chosen and why, and the alternatives rejected. The why is the point: it's what
  stops someone from undoing the choice later.
- **Fails**: symptom (error, crash step, wrong number), cause, fix, and which runs it affected. Keep failed
  commands that teach something.
- **Results**: verified numbers with their conditions (split, n, seed, metric name, epoch selection), as a
  table when there are several. Say what is still running or unverified.
- **Impact / next**: runs superseded, deleted or to rerun; open questions.

Only write what you saw in tool output or the conversation; mark estimates as such. When a new fact
contradicts an old claim, don't rewrite the old text: add `Superseded (YYYY-MM-DD): ...` next to it and
point to the new section.

Links: repo files relative to the page (`../../../graph_inject/README.md`), other wiki pages bundle-relative
(`/pages/hrm-graph/<page>.md`).

## 3. Housekeeping

- Set `last_updated` in the page's frontmatter.
- New page: add a line to `wiki/pages/hrm-graph/index.md` (`* [Title](file.md) - one-line summary.`).
- Substantial result or new page: append one line to the end of `wiki/log.md`
  (`- YYYY-MM-DD: [Page](pages/hrm-graph/<page>.md): <one-line finding>.`).
- If CLAUDE.md points to the topic and a key fact changed (a command, a path), update that pointer too.
- Run `.venv/bin/python scripts/validate_okf.py wiki` from the repo root; fix any error it reports for the
  pages you touched (the existing warnings on other pages are not yours).

Reply with the page path, the section added and a 1-2 line summary of what was recorded.
