---
type: Technical Reference
title: Mimir app appearance and settings
description: Cross-platform text scaling and settings organization.
tags: [mimir, app, appearance, settings]
status: draft
last_updated: 2026-10-17
confidence: high
---

# Mimir app appearance and settings

Moved from [the app runbook](mimir-app.md) on 2026-10-17 without changing the recorded behaviour.

## App text scaling (2026-09-24)

Flutter Settings offers 75–200% text sizing (5% steps, default/reset 100%).
`ChatStore.textScale` persists in the local archive independently of model
settings; invalid/missing values default to 100%. The root MediaQuery wraps the
Navigator, including open dialogs, and composes with nonlinear system scaling.
No native inference or network work is triggered. Native OS dialogs use system
sizing. Tests cover persistence, invalid values, nonlinear scaling and live
composer/dialog updates at phone and desktop widths with system enlargement.

Settings tabs: Model, Appearance, Online and Advanced. Advanced holds compaction
and profiles; Online holds search/feedback. Dependent controls hide when their
feature is off, preserving values; temporary busy locks stay disabled in place.
The model status describes acceleration; backend details sit by its selector.

