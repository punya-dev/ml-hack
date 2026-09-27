# ML Challenge 2026: Exploratory Data Analysis & Noise Audit Report

## 1. Dataset Scale & Country Breakdown

| Dataset | Split | Total Records | US (%) | India (%) | France (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Source 1** | Train | 2,206,821 | 1,323,633 (59.98%) | 883,188 (40.02%) | — |
| **Source 2** | Train | 5,034,616 | 3,016,817 (59.92%) | 2,017,799 (40.08%) | — |
| **Source 3** | Train | 5,285,603 | 3,170,056 (59.98%) | 2,115,547 (40.02%) | — |
| **Source 1** | Test  | 1,732,544 | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) |
| **Source 2** | Test  | 4,887,273 | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) |
| **Source 3** | Test  | 5,082,316 | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) |

**Key Findings:**
1. Training data contains strictly `US` (~60%) and `India` (~40%).
2. Test data introduces `France`, accounting for exactly **~14.4%–15.0%** of entities across all three sources.
3. String casing for country is 100% clean and consistent: `"US"`, `"India"`, `"France"` (no `"United States"`, `"USA"`, or trailing spaces).

---

## 2. Null & Missing Value Audit

| Column | Source 1 (Train / Test) | Source 2 (Train / Test) | Source 3 (Train / Test) |
| :--- | :--- | :--- | :--- |
| `entity_id` | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) |
| `business_name` | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) |
| `business_address`| 0 (0.00%) / 0 (0.00%) | 168,967 (3.36%) / 129,408 (2.65%) | 175,916 (3.33%) / 136,098 (2.68%) |
| `country` | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) | 0 (0.00%) / 0 (0.00%) |

**Postal / ZIP Code Availability in Addresses:**
- **US Addresses**: ~10.2% contain a 5-digit ZIP code. Most US addresses contain `Street, City, State` or `State, City, Street`.
- **Indian Addresses**: PIN codes are frequently omitted; locality/landmark names are dominant.

---

## 3. Name Length Distributions (Quantiles)

| Source | Min | 25% | 50% (Median) | 75% | 99% | Max | Median Tokens | Max Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Source 1** | 3 chars | 18 chars | 24 chars | 30 chars | 42 chars | 71 chars | 4 tokens | 12 tokens |
| **Source 2** | 2 chars | 18 chars | 25 chars | 31 chars | 48 chars | 104 chars| 4 tokens | 15 tokens |
| **Source 3** | 2 chars | 18 chars | 25 chars | 31 chars | 50 chars | 80 chars | 4 tokens | 13 tokens |

*Takeaway:* Business names are compact (median 24-25 characters, 4 words). Outliers up to 104 characters are typically DBA names or legal extensions (e.g., `Miranex formerly known as Sunrise Services Private Limited`).

---

## 4. Ground-Truth Match Pair Audit (Real Evidence)

We evaluated all **172,839 true match pairs** in the validation ground truth:

### A. Country Hard-Filter Feasibility
- Country mismatches between true pairs: **0 out of 172,839 (0.0000%)**
- **Conclusion**: In ground truth, **every single true match pair is within the same country**. While we should still maintain a fallback for unseen countries, partitioning candidate blocking by `country` is extremely safe and cuts comparison complexity by ~60%.

### B. Singletons
- Exactly **123,247 out of 2,206,821** Source 1 entities (5.585%) have no matches.
- In validation: 2,792 singletons (5.584%). Correctly predicting empty match sets for singletons yields an automatic 1.0 per singleton.

---

## 5. Noise Patterns Discovered & Normalization Blueprint

### Pattern 1: Script Variations & Transliteration
- **Discovered:** Names and states appear in both Latin script and Indic scripts (Devanagari, Bengali):
  - Example: `Seven Solutions Pvt Ltd` ↔ `सेवन सॉल्यूशंस प्रा. लि.`
  - Example: `Sunrise Services Private Limited` ↔ `সানরাইজ সার্ভিসেস প্রাইভেট লিমিটেড`
  - Example: State `Maharashtra` ↔ `महाराष्ट्र`, `West Bengal` ↔ `পশ্চিমবঙ্গ`.
- **Solution:** Apply Unicode NFKC normalization + `unidecode` transliteration as the very first step in the pipeline.

### Pattern 2: Legal Suffix Inconsistencies & Abbreviations
- **Discovered:**
  - `Private Limited` ↔ `Pvt Ltd` ↔ `Pvt. Ltd.` ↔ `Private Ltd` ↔ `प्रा. लि.`
  - `LLC` ↔ `L.L.C.` ↔ `Limited Liability Company`
  - `Inc` ↔ `Incorporated` ↔ `Inc.`
  - `Corp` ↔ `Corporation`
  - `PC` ↔ `P.C.` (Professional Corporation)
  - `SARL`, `SASU`, `SAS`, `EURL` (in France test data)
- **Solution:** Bidirectional dictionary (`configs/name_abbreviations.json`) expanding all abbreviations to canonical forms, PLUS a `core_name` feature stripping all legal suffixes.

### Pattern 3: Position Reordering of Legal Suffixes
- **Discovered:** `LLC NORTH MARSHALL` ↔ `North Marshall LLC`.
- **Solution:** `sorted_tokens` key (alphabetically sorted lowercase tokens rejoined) completely neutralizes suffix positioning and word order transpositions.

### Pattern 4: Domain Names / Websites as Business Names
- **Discovered:** `northmarshall.com` ↔ `North Marshall LLC`, `regionalsterlingyhncom` ↔ `Regional Sterling Yhn Inc`.
- **Solution:** Strip common TLDs (`.com`, `.org`, `.net`, `.in`, `com`) and split concatenated words or match against stripped name.

### Pattern 5: Trade Names / DBA / Formerly Known As
- **Discovered:** `Miranex formerly known as Sunrise Services Private Limited` ↔ `Sunrise Services Private Limited`.
- **Solution:** Regex split on `DBA`, `t/a`, `formerly known as`, `f/k/a`, `a/k/a` and parentheses `(...)` to generate dual candidate strings.

### Pattern 6: Noise Characters & Brackets
- **Discovered:** `Regional Sterling [Yhn]`, `>> Lumyx Studios Llc`, `##233 Keyser Road`, `-- Holloway Peak`.
- **Solution:** Strip leading/trailing punctuation and decorative symbols (`#`, `>`, `<`, `[`, `]`, `-`, `~`, `*`).

### Pattern 7: Address Abbreviations & Landmark Triggers
- **US Street Types:** `Rd` ↔ `Road`, `St` ↔ `Street`, `Ave` ↔ `Avenue`, `Dr` ↔ `Drive`, `Ct` ↔ `Court`, `Blvd` ↔ `Boulevard`.
- **US Units:** `Unit APARTMENT G` ↔ `Apt G`, `Floor 1` ↔ `Fl 1` ↔ `Fl. 1`.
- **US States:** Full name vs 2-letter postal abbreviation (`Alabama` ↔ `AL`, `Utah` ↔ `UT`, `Washington` ↔ `WA`).
- **Indian Landmarks & Types:** `Near`, `Opp` / `Opposite`, `Behind` / `B/H`, `Beside`, `Next to`, `Plot No.`, `H.No` (House No), `KH NO.` (Khasra No), `P O` (Post Office), `Cross`, `Layout`, `Sector`.
- **French Street Words:** `Boulevard` / `BD`, `Rue`, `Avenue` / `AV`, `Chemin`, `Impasse`, `bis`, `ter`.

### Pattern 8: Address Component Inversions
- **Discovered:** `AL, Tuscaloosa, 2814 Cherry Street` ↔ `02814 CHERRY STREET, TUSCALOOSA, AL`.
- **Solution:** `sorted_address_tokens` and token-bag / n-gram Jaccard matching to make distance invariant to token ordering.
