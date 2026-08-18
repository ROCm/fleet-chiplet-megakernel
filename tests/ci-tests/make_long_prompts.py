"""Build prompt files whose *post-chat-template* token count hits a target.

The seq-len sweep needs a prompt of exactly N tokens, but demo.py runs every
prompt through the chat template, which adds a variable preamble. Tokenizing
N tokens of corpus text and passing it in therefore lands somewhere above N.
This binary-searches the corpus prefix so the final templated length is
<= target, and reports what it actually got.

    python3 tests/ci-tests/make_long_prompts.py --model-path $MODEL_PATH \
        --out-dir /tmp/prompts --targets 512 1024 2048 ...

Writes <out-dir>/prompt_<target>.txt plus a lengths.json manifest.
"""
import argparse
import json
import os


def templated_len(tok, text):
    if getattr(tok, "chat_template", None):
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)
        return len(tok(formatted, add_special_tokens=False)["input_ids"])
    return len(tok(text)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out-dir", default="/tmp/prompts")
    ap.add_argument("--targets", nargs="+", type=int, required=True)
    ap.add_argument("--corpus", default="wikitext2")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    if args.corpus == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="test")
        lines = [t.strip() for t in ds["text"]
                 if t.strip() and not t.strip().startswith("=")]
        text = "\n\n".join(lines)
    else:
        with open(args.corpus, encoding="utf-8") as f:
            text = f.read()

    # Repeat until the corpus is long enough for the biggest target. Wikitext-2
    # test is ~280k tokens, so only 128k+overhead needs this in practice.
    ids = tok(text, add_special_tokens=False)["input_ids"]
    need = max(args.targets) * 2
    while len(ids) < need:
        text = text + "\n\n" + text
        ids = tok(text, add_special_tokens=False)["input_ids"]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {}
    for target in args.targets:
        # Binary-search the token prefix whose templated length is <= target.
        lo, hi, best = 1, min(len(ids), target * 2), None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = tok.decode(ids[:mid])
            n = templated_len(tok, cand)
            if n <= target:
                best, lo = (mid, cand, n), mid + 1
            else:
                hi = mid - 1
        assert best is not None, f"cannot fit target {target}"
        _, cand, n = best
        path = os.path.join(args.out_dir, f"prompt_{target}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cand)
        manifest[str(target)] = {"path": path, "templated_tokens": n,
                                 "chars": len(cand)}
        print(f"target={target:7d}  actual={n:7d}  chars={len(cand):9d}  {path}")

    with open(os.path.join(args.out_dir, "lengths.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
