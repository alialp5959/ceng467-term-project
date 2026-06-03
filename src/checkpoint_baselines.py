import os
import pandas as pd
import torch
import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 100


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def translate_batch(texts, model_name, batch_size=8, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=model_name):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(DEVICE)
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=max_length, num_beams=4)
        outputs.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return outputs


def compute_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)


def main():
    os.makedirs("results", exist_ok=True)
    en_lines = read_lines("flores200_dataset/devtest/eng_Latn.devtest")
    tr_lines = read_lines("flores200_dataset/devtest/tur_Latn.devtest")

    data = pd.DataFrame({"tr": tr_lines[:N_SAMPLES], "en": en_lines[:N_SAMPLES]})
    tr_refs, en_refs = data["tr"].tolist(), data["en"].tolist()

    copy_tr_en_preds = data["tr"].tolist()
    copy_en_tr_preds = data["en"].tolist()
    pretrained_tr_en_preds = translate_batch(data["tr"].tolist(), "Helsinki-NLP/opus-mt-tr-en")
    pretrained_en_tr_preds = translate_batch(data["en"].tolist(), "Helsinki-NLP/opus-mt-tc-big-en-tr")

    rows = []
    for model, direction, preds, refs in [
        ("Copy Baseline", "TR->EN", copy_tr_en_preds, en_refs),
        ("Pretrained MT Reference (Helsinki-NLP/opus-mt-tr-en)", "TR->EN", pretrained_tr_en_preds, en_refs),
        ("Copy Baseline", "EN->TR", copy_en_tr_preds, tr_refs),
        ("Pretrained MT Reference (Helsinki-NLP/opus-mt-tc-big-en-tr)", "EN->TR", pretrained_en_tr_preds, tr_refs),
    ]:
        bleu, chrf = compute_metrics(preds, refs)
        rows.append({"model": model, "direction": direction, "BLEU": bleu, "chrF": chrf, "status": "Complete"})

    pd.DataFrame(rows).to_csv("results/checkpoint_results.csv", index=False)
    pd.DataFrame({
        "source_tr": data["tr"],
        "reference_en": data["en"],
        "copy_baseline_output": copy_tr_en_preds,
        "pretrained_mt_output": pretrained_tr_en_preds
    }).to_csv("results/sample_translations_tr_en.csv", index=False)
    pd.DataFrame({
        "source_en": data["en"],
        "reference_tr": data["tr"],
        "copy_baseline_output": copy_en_tr_preds,
        "pretrained_mt_output": pretrained_en_tr_preds
    }).to_csv("results/sample_translations_en_tr.csv", index=False)


if __name__ == "__main__":
    main()
