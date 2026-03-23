# pyre-ignore-all-errors
import csv
import sys
import pandas as pd # type: ignore # pyre-ignore
from pathlib import Path
from q4.lattice_builder import build_lattice, lattice_wer, standard_wer, normalize # type: ignore # pyre-ignore

def main():
    if len(sys.argv) < 3:
        print("Usage: python q4_runner.py <input_csv> <output_csv>")
        return

    in_csv = sys.argv[1]
    out_csv = sys.argv[2]

    df = pd.read_csv(in_csv)
    # the columns are segment_url_link, Human, Model H, Model i, Model k, Model l, Model m, Model n
    
    results = []
    
    for _, row in df.iterrows():
        segment_url = str(row['segment_url_link'])
        # if segment url is nan, skip
        if segment_url == "nan":
            continue
            
        reference = str(row['Human']).strip()
        model_outputs = {
            "Model H": str(row.get('Model H', '')).strip(),
            "Model i": str(row.get('Model i', '')).strip(),
            "Model k": str(row.get('Model k', '')).strip(),
            "Model l": str(row.get('Model l', '')).strip(),
            "Model m": str(row.get('Model m', '')).strip(),
            "Model n": str(row.get('Model n', '')).strip(),
        }
        
        # Remove empty model outputs
        model_outputs = {k: v for k, v in model_outputs.items() if v and v != "nan"}
        if not model_outputs or reference == "nan" or not reference:
            continue
            
        names = list(model_outputs.keys())
        outs = list(model_outputs.values())
        
        lattice, meta = build_lattice(outs, reference, model_names=names)
        ref_words = normalize(reference)
        
        for name, out in model_outputs.items():
            hyp = normalize(out)
            std = standard_wer(hyp, ref_words)
            lat, _ = lattice_wer(hyp, lattice)
            
            results.append({
                "segment": segment_url.split('/')[-1] if segment_url else "unknown",
                "reference": reference,
                "model": name,
                "prediction": out,
                "standard_wer_pct": round(std * 100, 2),
                "lattice_wer_pct": round(lat * 100, 2),
                "wer_reduction_pct": round((std - lat) * 100, 2)
            })
            
    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"Processed {len(df)} segments. Lattice WER results saved to {out_csv}")

if __name__ == "__main__":
    main()
