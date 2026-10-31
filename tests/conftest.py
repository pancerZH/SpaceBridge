from dataclasses import dataclass

import numpy as np
import pytest

from spacebridge import CalibrationBundle, make_split, sample_distractors


def l2(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


@dataclass
class World:
    """Two embedding spaces generated from one latent space, with separate query/doc cones."""

    E_src: np.ndarray       # (N, d_s) source index
    bundle: CalibrationBundle
    Y_query: np.ndarray     # (K, d_t) target query embeddings of the anchor proxies
    Y_doc: np.ndarray       # (K, d_t) target doc embeddings of the anchor docs
    Q_src: np.ndarray       # (n_test, d_s) held-out test queries, source encoder
    Q_tgt: np.ndarray       # (n_test, d_t) same queries, target encoder


def make_world(n_docs=3000, K=400, n_test=300, d_lat=32, d_s=48, d_t=40, noise=0.3, seed=0) -> World:
    rng = np.random.default_rng(seed)
    A_s = rng.standard_normal((d_lat, d_s)) / np.sqrt(d_lat)
    A_t = rng.standard_normal((d_lat, d_t)) / np.sqrt(d_lat)
    doc_off_s, q_off_s = rng.standard_normal(d_s), rng.standard_normal(d_s)
    doc_off_t, q_off_t = rng.standard_normal(d_t), rng.standard_normal(d_t)

    Z = rng.standard_normal((n_docs, d_lat))
    E_src = l2(Z @ A_s + doc_off_s)

    anchor_ids = rng.choice(n_docs, K, replace=False)
    Zq = Z[anchor_ids] + noise * rng.standard_normal((K, d_lat))
    test_docs = rng.choice(n_docs, n_test)
    Zt = Z[test_docs] + noise * rng.standard_normal((n_test, d_lat))

    train_idx, val_idx = make_split(K, seed=seed)
    bundle = CalibrationBundle(
        src_lang="en",
        anchor_ids=anchor_ids,
        anchor_doc_texts=[f"doc {i}" for i in anchor_ids],
        anchor_query_texts=[f"proxy {i}" for i in anchor_ids],
        X_doc_raw=E_src[anchor_ids],
        X_query_raw=l2(Zq @ A_s + q_off_s),
        train_idx=train_idx,
        val_idx=val_idx,
        distractor_ids=sample_distractors(n_docs, anchor_ids, seed=seed),
        mu_x_doc=E_src.mean(0),
    )
    return World(
        E_src=E_src,
        bundle=bundle,
        Y_query=l2(Zq @ A_t + q_off_t),
        Y_doc=l2(Z[anchor_ids] @ A_t + doc_off_t),
        Q_src=l2(Zt @ A_s + q_off_s),
        Q_tgt=l2(Zt @ A_t + q_off_t),
    )


def top_k(Q: np.ndarray, E: np.ndarray, k: int = 10) -> np.ndarray:
    return np.argsort(-(Q @ E.T), axis=1)[:, :k]


def overlap_at_k(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


@pytest.fixture(scope="session")
def world() -> World:
    return make_world()
