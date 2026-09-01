# Report: Preprocessing Pipeline on February 2025 Data (Infrabel)

Date of execution: September 1, 2026
Source raw file: C:\Users\claud\Downloads\Data_raw_punctuality_202502.csv (254.6 MB)

---

## 1. Executive Summary & Processed Outputs

The raw monthly punctuality export for February 2025 was processed through the Cortex junction extraction pipeline. The pipeline streamed 1,522,221 raw rows, filtered the 5 central Brussels tunnel stations, validated travel sequences, and generated two standardized working datasets:

| Output File Path | Format | Size | Rows | Columns | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data/junction/junction_traversing_202502.csv` | Long Form | 22.4 MB | 118,005 | 24 | Sequential, row-level (5 points per traversal) |
| `data/junction/traversals_202502.csv` | Wide Form | 8.3 MB | 23,601 | 56 | 1 row per train traversal, travel-ordered |

---

## 2. Raw Dataset Characteristics

- **Raw File:** `Data_raw_punctuality_202502.csv` (Infrabel Open Data)
- **Total Network Rows:** 1,522,221 records
- **Total Distinct Trips:** 26,149 trips
- **Format:** Comma-separated (`,`), UTF-8 encoded
- **Key Raw Attributes:**
  - `DATDEP`: Service date (e.g. `01FEB2025`)
  - `TRAIN_NO`: Train identifier (e.g. `3117`)
  - `RELATION`: Commercial service line (e.g. `IC 31`)
  - `PTCAR_NO`: Operational point identifier (Infrabel network code)
  - `LINE_NO_ARR` / `LINE_NO_DEP`: Inbound/outbound line/track numbers
  - `PLANNED_TIME_ARR` / `PLANNED_TIME_DEP`: Timetable scheduled time (`HH:MM:SS`)
  - `REAL_TIME_ARR` / `REAL_TIME_DEP`: Actual measured timestamp
  - `DELAY_ARR` / `DELAY_DEP`: Signed delay in seconds (negative = early, positive = late)
  - `THOP1_COD`: Stop category (`=` commercial stop, `D` pass-through, `P` terminus/origin)

---

## 3. Step-by-Step Preprocessing Logic

```
[Raw Infrabel Export (1.52M rows)]
               │
               ▼
[Step 1: Spatial Filter (PTCARs: 215, 216, 217, 220, 221)]
               │
               ▼
[Step 2: Group by Trip Key (DATDEP, TRAIN_NO) -> 26,149 trips]
               │
               ▼
[Step 3: Traversal Integrity Check]
   ├── Contains both Nord (221) and Midi (220)?
   ├── Exactly 5 station observations?
   └── Follows strict physical tunnel order?
         ├── S2N: 220 -> 217 -> 215 -> 216 -> 221
         └── N2S: 221 -> 216 -> 215 -> 217 -> 220
               │
               ├── Dropped: 2,548 single-station/terminating trips
               └── Retained: 23,601 valid full traversals (100% regular)
               │
               ▼
[Step 4: Track Identification & Metric Derivation]
   ├── Physical Tunnel Track: LINE_NO_DEP at Central (0/1 to 0/6)
   ├── Ordered Points p1 (entry) -> p5 (exit)
   ├── entry_delay = p1 departure delay
   ├── exit_delay = p5 arrival delay
   └── delay_gained = exit_delay - entry_delay
               │
               ▼
[Step 5: File Serialization]
   ├── data/junction/junction_traversing_202502.csv (Long form)
   └── data/junction/traversals_202502.csv (Wide form)
```

### 3.1 Spatial Filtering
Only records belonging to the 5 operational points of the Brussels North-South tunnel are extracted:
- `221`: Bruxelles-Nord / Brussel-Noord
- `216`: Bruxelles-Congrès / Brussel-Congres
- `215`: Bruxelles-Central / Brussel-Centraal
- `217`: Bruxelles-Chapelle / Brussel-Kapellekerk
- `220`: Bruxelles-Midi / Brussel-Zuid

### 3.2 Traversal Filtering & Direction Recovery
Trips are grouped by `(DATDEP, TRAIN_NO)`. A trip is kept only if it completely traverses the junction (touches both boundary stations `220` and `221`).
- Trains only touching Midi or Nord without traversing (2,548 trips, primarily terminating at Midi) are excluded.
- Direction is resolved from station sequence:
  - **S2N (South to North):** `220 → 217 → 215 → 216 → 221`
  - **N2S (North to South):** `221 → 216 → 215 → 217 → 220`
- Zero trips failed sequence order (100% regularity).

### 3.3 Tunnel Track Assignment
At Bruxelles-Central (`215`), the column `LINE_NO_DEP` uniquely identifies the physical tunnel track:
- **Tracks `0/1`, `0/3`, `0/5`:** South-to-North lines
- **Tracks `0/2`, `0/4`, `0/6`:** North-to-South lines

### 3.4 Feature Transformation Along Direction of Travel
Variables are mapped to travel order ($p_1, p_2, p_3, p_4, p_5$):
- $p_1$: Entry station (Midi for S2N, Nord for N2S)
- $p_2$: Congrès (N2S) or Chapelle (S2N)
- $p_3$: Bruxelles-Central (bottleneck station for both directions)
- $p_4$: Chapelle (N2S) or Congrès (S2N)
- $p_5$: Exit station (Nord for S2N, Midi for N2S)

Key propagation metrics:
- `entry_delay` = `p1_delay_dep` (departure delay into junction)
- `exit_delay` = `p5_delay_arr` (arrival delay leaving junction)
- `delay_gained` = `exit_delay - entry_delay` (net seconds gained/lost in tunnel)

---

## 4. Summary of Preprocessed February Data

- **Total Traversals:** 23,601
- **Direction Split:**
  - N2S: ~11,830 traversals
  - S2N: ~11,771 traversals
- **Integrity:** Complete, validated, and ready for causal discovery and time-series modeling.
