# Data Quality Platform

A lightweight, browser-based data quality tool built with Python and Streamlit. Upload any tabular dataset to profile, validate, clean, and export — no ML models or LLMs required.

## Features

### Data Sources
- **CSV** file upload
- **Excel** file upload (`.xlsx`, `.xls`)
- **Google Sheets** via public URL

### Tabs

| Tab | Description |
|-----|-------------|
| **Overview** | Dataset preview, row/column counts, completeness score, duplicate count, column type analysis |
| **Profiling Report** | Auto-generated HTML report via ydata-profiling (minimal or full mode) |
| **Quality Checks** | Missing values, duplicates, data types, outlier detection (IQR + Z-score), distribution stats, uniqueness/cardinality |
| **MLoS QA** | 18 domain-specific validation rules for MLoS settlement data with cross-table checks against a takeoffpoint reference table |
| **Standardization** | Z-score normalization, min-max scaling, whitespace trimming, case conversion, date normalization, duplicate/missing row handling, value imputation |
| **Download** | Export cleaned data as CSV or Excel |

### MLoS Domain Validation Rules

The MLoS QA tab validates settlement data against these rules:

| # | Rule |
|---|------|
| 1 | All wards must have a takeoffpoint in the takeoffpoint table |
| 2 | `takeoffpoint` in MLoS must match `name` in takeoffpoint table |
| 3 | `takeoffpoint_code` in MLoS must match `code` in takeoffpoint table |
| 4 | `ward_code` in MLoS must match `wardcode` in takeoffpoint table |
| 5 | `security_compromised` = Y or N, no nulls |
| 6 | `accessibility_status` = Fully Accessible, Partially Accessible, or Inaccessible |
| 7 | Partially Accessible / Inaccessible must have a reason |
| 8 | `habitation_status` = Abandoned, Migrated, Inhabited, or Partially Inhabited |
| 9 | `set_pop`, `target_pop`, `no_of_houses`, `noncpliant` must not be null |
| 10 | `team_code` = number or NA |
| 11 | `day_of_activity` = valid day combination (1, 1_2, 1_2_3, 1_2_3_4, 2, 2_3, 2_3_4, 3, 3_4, 4) or NA |
| 12 | `urban`, `rural`, `scattered` = Y or N; can't be both urban and rural; urban can't be scattered |
| 13 | Profile columns = Y, N, or NA |
| 14 | `source` = MLoS |
| 15 | `editer` = firstname.surname format, all lowercase |
| 16 | `globalid` = valid UUID |
| 17 | `fc_globalid` = valid UUID (auto-generate available) |
| 18 | `settlementarea_globlid` = valid UUID (auto-generate available; planned for removal) |

Rules 1-4 require uploading a takeoffpoint reference table. Rules 5-18 run on the MLoS table alone.

## Project Structure

```
data_quality_platform/
├── app.py                    # Streamlit application (main entry point)
├── requirements.txt          # Python dependencies
├── README.md
└── utils/
    ├── __init__.py
    ├── data_loader.py        # CSV, Excel, Google Sheets loaders
    ├── qa_checks.py          # Generic QA checks (missing values, duplicates, outliers, etc.)
    ├── mlos_qa_checks.py     # MLoS domain-specific validation rules
    └── standardization.py    # Data cleaning and transformation functions
```

## Setup

### Requirements
- Python 3.9+

### Installation

```bash
# Clone or download the project
cd data_quality_platform

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Upload** your dataset from the sidebar (CSV, Excel, or Google Sheet URL)
2. **Overview** tab shows a quick summary of your data
3. **Profiling Report** tab generates a detailed ydata-profiling report
4. **Quality Checks** tab lets you run generic QA checks on any dataset
5. **MLoS QA** tab runs the 18 domain-specific validation rules (optionally upload a takeoffpoint table for cross-table checks)
6. **Standardization** tab lets you clean and transform columns
7. **Download** tab exports your cleaned dataset as CSV or Excel

## Tech Stack

- **Streamlit** — interactive web UI
- **Pandas / NumPy / SciPy** — data manipulation and statistics
- **ydata-profiling** — automated profiling reports
- **openpyxl** — Excel file support
