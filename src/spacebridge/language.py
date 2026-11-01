"""Parallel-sentence banks for cross-lingual calibration (design §4.4, §7.1). Text is downloaded at runtime and cached
under the git-ignored ``data/`` directory; none of it ships with this package."""

import json
from pathlib import Path

FLORES_REPO = "openlanguagedata/flores_plus"   # gated (CC BY-SA 4.0): needs an HF token with accepted terms
BELEBELE_REPO = "facebook/belebele"             # its passages are FLORES-200 passages (CC BY-SA 4.0)
FLORES_CODES = {
    "en": "eng_Latn", "zh": "cmn_Hans", "de": "deu_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "ar": "arb_Arab",
    "vi": "vie_Latn", "hi": "hin_Deva", "tr": "tur_Latn", "el": "ell_Grek", "th": "tha_Thai", "ro": "ron_Latn",
}
FLORES_CACHE = Path("data/flores")
FLORES_SPLITS = ("dev", "devtest")
_FLORES_VERSION = 1  # bump when filtering changes


def load_flores(lang: str, belebele_filter: bool = True) -> dict[str, dict]:
    """Row-aligned human-translated (English, ``lang``) FLORES+ sentence pairs per split (design §4.4, §7.1).

    Rule 6 of §7.1: sentences whose English text occurs inside a Belebele passage are dropped
    (``belebele_filter``; Belebele evaluates on FLORES text), and so are sentences that occur in both
    ``dev`` and ``devtest``. Returns ``{split: {"en": [...], lang: [...], "ids": [...]}}`` plus
    ``"counts"`` (sentences before / after each filter). Downloads at runtime and caches only the filtered
    pairs under the git-ignored ``data/flores/``; raises if the gated repo is not accessible.
    """
    tag = "belebele-filtered" if belebele_filter else "unfiltered"
    path = FLORES_CACHE / f"flores_en-{lang}_{tag}_v{_FLORES_VERSION}.json"
    if path.exists():
        return json.loads(path.read_text())
    from huggingface_hub import hf_hub_download

    def rows(split: str, code: str) -> list[dict]:
        local = hf_hub_download(FLORES_REPO, f"{split}/{code}.jsonl", repo_type="dataset")
        return [json.loads(line) for line in Path(local).open()]

    norm = lambda t: " ".join(t.split())
    raw = {}
    for split in FLORES_SPLITS:
        en, xx = rows(split, FLORES_CODES["en"]), rows(split, FLORES_CODES[lang])
        if [r["id"] for r in en] != [r["id"] for r in xx]:
            raise ValueError(f"FLORES {split} en / {lang} are not row-aligned")
        raw[split] = (en, xx)
    passages: list[str] = []
    if belebele_filter:
        local = hf_hub_download(BELEBELE_REPO, f"data/{FLORES_CODES['en']}.jsonl", repo_type="dataset")
        passages = sorted({norm(json.loads(line)["flores_passage"]) for line in Path(local).open()})
    texts = {s: {norm(r["text"]) for pair in raw[s] for r in pair} for s in FLORES_SPLITS}
    both = texts["dev"] & texts["devtest"]
    out: dict[str, dict] = {"counts": {}}
    for split, (en, xx) in raw.items():
        keep = [
            i for i in range(len(en))
            if norm(en[i]["text"]) not in both and norm(xx[i]["text"]) not in both
            and not any(norm(en[i]["text"]) in p for p in passages)
        ]
        out[split] = {"ids": [en[i]["id"] for i in keep], "en": [en[i]["text"] for i in keep], lang: [xx[i]["text"] for i in keep]}
        out["counts"][split] = {"raw": len(en), "kept": len(keep)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False))
    return out


def flores_available(lang: str) -> bool:
    """Whether FLORES pairs for ``lang`` are cached or downloadable (the gated repo is accessible)."""
    if lang not in FLORES_CODES:
        return False
    try:
        load_flores(lang)
    except Exception:  # no token, terms not accepted, offline
        return False
    return True


SHIFT_BANKS = ("auto", "flores", "flores-unfiltered", "tatoeba")


def load_shift_bank(query_lang: str, n: int = 256, bank: str = "auto") -> tuple[list[str], list[str], str]:
    """Parallel (English, query-language) sentences for the language-cone shift (design §4.4), and the bank's name.

    ``auto``: the Belebele-filtered FLORES ``dev`` split when the gated repo is accessible (at most ``n`` pairs;
    the filter leaves 94 of 997 for zh, all of them used), else the first ``n`` human-translated Tatoeba pairs
    from ``mteb/tatoeba-bitext-mining``. ``flores-unfiltered`` (ablation) skips the Belebele filter.
    """
    if bank == "auto":
        bank = "flores" if flores_available(query_lang) else "tatoeba"
    if bank in ("flores", "flores-unfiltered"):
        dev = load_flores(query_lang, belebele_filter=bank == "flores")["dev"]
        return dev["en"][:n], dev[query_lang][:n], bank
    from datasets import load_dataset

    config = {"zh": "cmn-eng"}[query_lang]
    ds = load_dataset("mteb/tatoeba-bitext-mining", config, split="test")
    pairs = [(e.strip(), q.strip()) for q, e in zip(ds["sentence1"], ds["sentence2"])][:n]
    return [e for e, _ in pairs], [q for _, q in pairs], "tatoeba"
