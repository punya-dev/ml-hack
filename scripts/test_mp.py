import time
import pandas as pd
from multiprocessing import Pool
from src.name_normalizer import normalize_business_name
from src.address_parser import normalize_business_address
from src.country_normalizer import normalize_country


def process_chunk(chunk_rows):
    # Returns counts for each metric
    res = []
    for name, addr, ctry in chunk_rows:
        c = normalize_country(ctry)
        pn = normalize_business_name(name)
        pa = normalize_business_address(addr, country=c)
        res.append((
            c,
            bool(pn["clean_name"]),
            bool(pn["core_name"]),
            bool(pn["sorted_tokens"]),
            bool(pn["phonetic_key"]),
            bool(pn["acronym_key"]),
            bool(pa["street"]),
            bool(pa["city"]),
            bool(pa["state"]),
            bool(pa["postal_code"]),
            bool(pa["landmark_text"]),
            bool(pa["sorted_address_tokens"])
        ))
    return res


def main():
    df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", nrows=50000)
    rows = list(zip(df["business_name"].fillna(""), df["business_address"].fillna(""), df["country"].fillna("")))
    batch_size = 5000
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    t0 = time.time()
    with Pool(8) as pool:
        results = pool.map(process_chunk, batches)
    el = time.time() - t0
    total = sum(len(r) for r in results)
    print(f"Processed {total} rows across 8 workers in {el:.2f}s ({total/el:.0f} rows/sec)!")


if __name__ == "__main__":
    main()
