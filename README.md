# Przewidywanie defaultu firm — model interpretowalny vs black-box


## Struktura repozytorium

```
proj1/
├── data/
│   ├── zbiór_4.csv                  # surowe dane (3000 × 223)
│   ├── development_set.csv          # zbiór deweloperski: 2401 wierszy, 144 defaulty, kolumna cv_fold (0–4)
│   ├── test_holdout_set.csv         # odłożony zbiór testowy: 599 wierszy, 36 defaultów
│   └── selected_features.json       # listy cech z etapu EDA (top 20, cechy zachowane i odrzucone)
├── eda_and_data_separation.ipynb    # EDA, grupowanie firm, podział train/test i 5 foldów, IV/WoE
├── interpretable_model.ipynb        # karta scoringowa WoE: trening, metryki, interpretacja,
│                                    # stabilność, kalibracja, 4%, progi, ratingi; eksport do artifacts/
├── black_box_analysis.ipynb         # XGBoost: strojenie, SHAP, PDP/ICE/ALE, LIME, kalibracja, 4%,
│                                    # progi, ratingi, kontrfakty, porównanie z kartą scoringową
├── MODEL_CARD.md                    # model card obu modeli: cel, dane, ograniczenia, ryzyka,
│                                    # wyjaśnienia, kalibracja, plan monitoringu
└── requirements.txt
```

---

## Instalacja

Wymagany jest Python ≥ 3.11.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Uruchomienie

Notebooki uruchamiamy z katalogu głównego repozytorium (`proj1/`), w poniższej kolejności:

1. **`eda_and_data_separation.ipynb`** — opcjonalny, bo gotowe podziały są już w `data/`.
   Notebook wczytuje `zbiór_4.csv` z bieżącego katalogu, więc przed uruchomieniem zmień `DATA_PATH` na `data/zbiór_4.csv` (albo uruchom notebook z katalogu `data/`). Notebook nie nadpisuje plików w `data/`, tylko odtwarza analizę i podział (`random_state=42`).
2. **`interpretable_model.ipynb`** — wczytuje `data/development_set.csv` i `data/test_holdout_set.csv`.
3. **`black_box_analysis.ipynb`** — wczytuje te same zbiory. W części porównawczej korzysta z plików w `artifacts/` (tworzy się po kroku 2), więc **musi być uruchomiony po kroku 2**. Strojenie (`RandomizedSearchCV`, 50 × 4 dopasowań) trwa kilka minut.

Przykład uruchomienia bez interfejsu:

```bash
jupyter nbconvert --to notebook --execute --inplace interpretable_model.ipynb
jupyter nbconvert --to notebook --execute --inplace black_box_analysis.ipynb
```


