#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def excess_loss_scores(weak_by_q: dict, strong_by_q: dict, n_options: dict) -> tuple[dict, dict]:
    """Per-question excess loss, loss_strong - loss_weak at the weak model's own pick.

    weak_by_q / strong_by_q: {source_id: [P(correct) of each option]} per question, from
    the weak and the untuned strong model. A list may repeat its block of options once
    per row of the question; only the first n_options[source_id] entries are read. Each
    model's block is normalized into an option distribution q, the pick is
    yhat = argmax q_w, and the score is log q_w[yhat] - log q_s[yhat]. No gold label is read.

    Returns ({source_id: score}, {source_id: yhat}).
    """
    elloss_by_q, pick_by_q = {}, {}
    for src in n_options:
        K = max(n_options[src], 1)
        qw = np.clip(np.asarray(weak_by_q[src][:K], dtype=np.float64), 1e-12, None)
        qs = np.clip(np.asarray(strong_by_q[src][:K], dtype=np.float64), 1e-12, None)
        qw, qs = qw / qw.sum(), qs / qs.sum()
        yhat = int(np.argmax(qw))
        pick_by_q[src] = yhat
        elloss_by_q[src] = float(np.log(qw[yhat]) - np.log(qs[yhat]))
    return elloss_by_q, pick_by_q
