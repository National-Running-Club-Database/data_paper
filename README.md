# Data Paper

Analysis code for the [National Running Club Database public dataset](https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset): all NRCD **public, approved** release data with **PII removed** (not limited to cross country).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

The upstream repo holds the organization’s full public approved export—anonymized tables for meets, athletes, teams, events, results, course details, and related fields across sports and event types. CSV files are not committed to this repo (`data/` is gitignored). Pull them from the public dataset repository into `data/`:

```bash
mkdir -p data

# Shallow clone, copy CSVs, remove temp clone
git clone --depth 1 \
  https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset.git \
  .tmp_nrcd_dataset
cp .tmp_nrcd_dataset/*.csv data/
rm -rf .tmp_nrcd_dataset
```

Expected files under `data/` (from the [dataset README](https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset/blob/main/README.md)):

| File | Description |
|------|-------------|
| `athlete.csv` | Athlete info (PII removed) |
| `athlete_team_association.csv` | Athlete–team links |
| `course_details.csv` | Course and weather |
| `joined.csv` | Denormalized join of all tables |
| `meet.csv` | Meet metadata |
| `result.csv` | Individual race results |
| `running_event.csv` | Event definitions |
| `sport.csv` | Sport metadata |
| `team.csv` | Team metadata |

`scripts/utils.py` reads at least `result.csv`, `meet.csv`, `athlete.csv`, and `running_event.csv` from `data/`.

### Alternative: download individual files

If you prefer not to clone the repo:

```bash
mkdir -p data
BASE="https://raw.githubusercontent.com/National-Running-Club-Database/national_running_club_database_public_dataset/main"
for f in athlete athlete_team_association course_details joined meet result running_event sport team; do
  curl -fsSL -o "data/${f}.csv" "${BASE}/${f}.csv"
done
```

## Project layout

```
data/              # CSVs (local only, not in git)
scripts/
  utils.py         # Time parsing, course lookup, data loading helpers
requirements.txt
```

## Dataset citation

When using this data, cite the [National Running Club Database public dataset](https://github.com/National-Running-Club-Database/national_running_club_database_public_dataset) and follow its usage terms (public approved release; PII removed). Contact and author details are listed in that repository’s README and on the [NRCD organization](https://github.com/National-Running-Club-Database).
