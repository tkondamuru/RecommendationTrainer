# Data Preprocessing Workflow

This document explains how raw sales data is transformed into high-precision training sets for the ML.NET Recommendation Engine using the [process_baskets.py](file:///c:/Development/labs/marketanalysis/RecommendationTrainer/process_baskets.py) script.

---

## Core Reference Files

*   **all_sundries.csv**: Master catalog of accessories containing internal IDs, part numbers, and detailed human-readable descriptions.
*   **autoparts.csv**: Master catalog of primary glass parts used to populate the autosuggest search dropdown.
*   **interchanges.csv**: Mapping file for normalizing functionally identical accessories (e.g., different brands of urethane) into single "Master IDs."
*   **part_vehicles.tsv**: Tab-separated dataset mapping glass parts to their specific vehicle year/make/model fitment strings.
*   **sales_history_raw.csv**: The source transaction log containing historical market baskets used to train the recommendation engine.

---

## 1. Input Data Formats

### Raw Sales History (`DataExtractor/sales_history_raw_sample.csv`)
The primary source of truth, containing transaction headers and line items.
*   `LOC_NO`: Warehouse location (used for regional splitting).
*   `SHIPPER_NO`: Unique transaction ID (used for basket grouping).
*   `ITEM_UID_NO`: The part ID.
*   `CATGRY_CD`: Used to distinguish between **Glass** (< 60) and **Sundries** (>= 60).

### Interchanges (`DataExtractor/interchanges_sample.csv`)
A mapping file used to handle **Sundry Part Equivalency**.
*   **Purpose:** If two different brands of Urethane or two different Clip SKUs are functionally identical, they are normalized to a single "Master ID."
*   **Effect:** This prevents "Data Sparsity" for accessories. It ensures that the model learns the relationship between "Glass X" and "Urethane Type Y" regardless of which brand was actually sold.

---

## 2. The Preprocessing Pipeline

### Step 1: Normalization
The script first loads the interchange map and replaces all part IDs with their **Master ID**. This ensures that the engine learns patterns for the "part type" rather than a specific brand SKU.

### Step 2: Basket Grouping
Items are grouped by `ORDER_KEY` (a combination of `LOC_NO` and `SHIPPER_NO`).
*   An order is considered a **Valid Basket** only if it contains at least one **Glass** part AND at least one **Sundry** part.
*   Orders with only glass or only sundries are discarded for training purposes (though they might be used later for trend analysis).

### Step 3: Regional Splitting
The data is split based on `LOC_NO`:
*   **US:** Locations 100 - 899.
*   **CA:** Locations 900 - 999.
*   **Rationale:** Market preferences and inventory availability differ significantly between the US and Canada. Training separate models prevents "Cross-Border Pollution."

### Step 4: Positive Pair Extraction
For every valid basket, the script generates a "Positive Pair" (Label 1.0) for every combination of glass and sundry found in that order.

### Step 5: Aggressive Negative Sampling (1:10)
To teach the model **Discrimination**, the script generates 10 "Negative Pairs" (Label 0.0) for every 1 positive pair.
*   **Method:** For a given Glass ID, the script picks 10 random Sundry IDs that were **never** bought with that glass.
*   **Popularity Weighting:** Negatives are sampled based on the popularity of the sundry. This teaches the model: *"Just because this Urethane is popular doesn't mean it goes with this specific Side Window."*

---

## 3. Output Training Files
The pipeline produces two clean CSV files ready for `RecommendationModelTrainer.cs`:
*   `recommendation_training_US.csv`
*   `recommendation_training_CA.csv`

Each file contains three columns: `glass_id`, `sundry_id`, and `LABEL`.
