# Model Card — przewidywanie defaultu firm (`zbiór_4.csv`)

Dokument opisuje dwa modele PD (prawdopodobieństwa defaultu) zbudowane w projekcie:

| | Model interpretowalny | Model black-box |
| :--- | :--- | :--- |
| Algorytm | Regresja logistyczna na zmiennych WoE (karta scoringowa) | XGBoost (gradient boosting drzew) |
| Liczba cech | 8 | 109 |
| Notebook | `interpretable_model.ipynb` | `black_box_analysis.ipynb` |
| Kalibracja | Platt + przesunięcie logitowe do PD 4% | Platt + przesunięcie logitowe do PD 4% |
| Wyjaśnienia | Wbudowane (współczynniki, punkty scoringowe) | Post-hoc (SHAP, PDP/ICE/ALE, LIME, kontrfakty) |

---

## 1. Cel i zastosowanie

**Cel:** oszacowanie prawdopodobieństwa, że firma zbankrutuje w horyzoncie około roku, na podstawie danych z jej sprawozdania finansowego.

**Zastosowania:**
- decyzja kredytowa (przyjęcie lub odrzucenie wniosku na podstawie progu PD),
- przypisanie klasy ratingowej (AAA–CCC),
- wsparcie wyceny ryzyka i rezerw (PD skalibrowane do 4%).

**Poza zakresem:** osoby fizyczne, firmy spoza populacji z próby (inne branże lub okresy), automatyczna odmowa bez udziału analityka, wyliczanie wymogów kapitałowych bez dodatkowej walidacji regulacyjnej.

**Użytkownicy:** analitycy kredytowi, zespół ryzyka, walidacja modeli.

---

## 2. Dane (wspólne dla obu modeli)

- **Źródło:** `zbiór_4.csv` zawiera 3000 sprawozdań finansowych firm i 223 kolumny, w tym zmienną objaśnianą `default`.
- **Klasy:** 2820 firm bez defaultu (94%) i 180 z defaultem (6%).
- **Ochrona przed wyciekiem:**
  - **Grupowanie firm:** zbiór nie ma identyfikatora firmy, więc rekordy tej samej firmy z różnych lat łączymy heurystycznie (ten sam kod PKD i forma własności, wiek firmy różniący się o wielokrotność około 365 dni). Wychodzi 2513 grup, w tym 350 firm z kilkoma sprawozdaniami.
  - **Kolumny stałe:** usunięto 3 kolumny o zerowej wariancji.
- **Podział** (stratyfikowany względem defaultu i grupowy, więc wszystkie rekordy firmy trafiają do jednego zbioru):

| Zbiór | Obserwacje | Defaulty |
| :--- | ---: | ---: |
| Deweloperski (5 foldów CV) | 2401 | 144 (6,00%) |
| Testowy (odłożony) | 599 | 36 (6,01%) |

- **Jakość danych:**
  - **Braki:** usunięto 7 cech z ponad 90% braków (głównie przepływy pieniężne).
  - **Wartości nieskończone:** występują w 3 cechach (dzielenie przez zero).
  - **Przesiew IV:** 42 cechy mają IV < 0,02.
- **Wykorzystanie danych przez modele:**
  - **Model interpretowalny:** uczony na 5 foldach zbioru deweloperskiego. Kalibrator i przesunięcie do 4% wyznaczone na predykcjach out-of-fold (OOF) całego zbioru deweloperskiego.
  - **Black-box:** uczony na foldach 0–3 (1921 obserwacji, 115 defaultów). Fold 4 (480 obserwacji, 29 defaultów) jest odłożony do kalibracji i wyboru progu.

---

## 3. Model interpretowalny — karta scoringowa WoE

### 3.1 Opis modelu
- **Cechy:**
  - rentowność: `wsk_ROA`,
  - płynność: `wsk_pokrycie_zob_kr_gotowkowe_1` i `wsk_poziom_kapitalu_obrotowego_netto`,
  - zadłużenie: `wsk_ogolnego_zadluzenia_1` i `Kapital_wlasny`,
  - aktywność: `wsk_rotacja_aktywow_1`,
  - wiek firmy: `wsk_liczba_dni_istnienia`,
  - dodatkowo `wsk_index_nowe`.

  Wszystkie VIF są poniżej 1,5.
- **Transformacja WoE:** każda cecha ma 4 przedziały kwantylowe.
- **Regresja:** logistyczna bez ważenia klas. Wariant z balansowaniem nie poprawił AUC (0,777 wobec 0,780), a zawyżał średnie PD do 37%.
- **Skala punktowa:** 600 pkt odpowiada szansom 50:1, PDO = 20.

### 3.2 Wyniki

| Zbiór | ROC-AUC | PR-AUC | KS | Log-loss | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: |
| OOF (dev) | 0,780 | 0,189 | 43,1% | 0,197 | 0,052 |
| Test | 0,790 | 0,173 | 48,5% | 0,198 | 0,053 |
| Model bez wiedzy (stałe 6%) | 0,500 | 0,060 | 0% | 0,227 | 0,056 |

Między foldami ROC-AUC waha się od 0,71 do 0,88, co wynika z tego, że na fold przypada tylko około 29 defaultów.

### 3.3 Wyjaśnienia globalne
- **Znaki współczynników:** wszystkie β są ujemne. Wyższe WoE oznacza bezpieczniejszy przedział, więc przesunięcie firmy do bezpieczniejszego przedziału dowolnej cechy obniża PD.
- **Najsilniejsze cechy** (β, w nawiasie iloraz szans): `wsk_index_nowe` −0,93 (0,39), `wsk_ogolnego_zadluzenia_1` −0,63, `Kapital_wlasny` −0,60, `wsk_ROA` −0,51, `wsk_poziom_kapitalu_obrotowego_netto` −0,28.
- **Cechy do usunięcia:** `wsk_rotacja_aktywow_1`, `wsk_liczba_dni_istnienia` i `wsk_pokrycie_zob_kr_gotowkowe_1` mają β bliskie 0. Dwie ostatnie dodatkowo **zmieniają znak między foldami**. Model bez tych trzech cech daje praktycznie te same wyniki (AUC OOF 0,783, test 0,789).
- **PDP/ICE:** krzywe mają kształt schodków na granicach przedziałów WoE i są równoległe, bo model nie zawiera interakcji.

### 3.4 Wyjaśnienia lokalne
- **Rozkład punktów:** wynik firmy to Offset plus suma punktów z każdej cechy. Rozkład jest **dokładny**, bez przybliżeń, więc każdą decyzję można uzasadnić listą cech, które dały firmie najmniej punktów.
- **Przykłady z testu:**
  - Firma zdrowa (PD 0,1%) zbiera punkty na `wsk_index_nowe`, kapitale własnym i ROA.
  - Firma, która zbankrutowała (PD 33%), traci punkty na `wsk_index_nowe` i kapitale własnym.

### 3.5 Kalibracja i dopasowanie do 4%

| Zbiór | Obs. DR | Śr. PD (surowe) | Śr. PD (Platt) | Śr. PD (4%) | CITL (Platt) | Nachylenie (Platt) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| OOF | 6,00% | 5,99% | 6,03% | 4,00% | −0,006 | 0,94 |
| Test | 6,01% | 6,88% | 6,78% | 4,51% | −0,139 | 1,03 |

- **Kalibracja na starcie:** model bez balansowania jest dobrze skalibrowany od początku (ECE OOF 0,008).
- **Wybór metody:** Platt zmienia predykcje niewiele. Regresja izotoniczna przeuczała się (AUC OOF 0,758, nachylenie 0,40), więc jej nie stosujemy.
- **Przesunięcie do 4%:** δ = −0,462 na skali logitowej, czyli szanse defaultu mnożone przez 0,63. Ranking firm się nie zmienia (AUC, KS i PR-AUC bez zmian).

### 3.6 Decyzje i ratingi
- **Próg decyzyjny:** przy stosunku kosztów FN:FP = 10:1 wynosi t = 9,1% na PD po Platcie, co odpowiada 5,9% na skali 4%.
  - Na teście model odrzuca 28% firm, wykrywa 67% defaultów i obniża koszt o 26% względem akceptowania wszystkich.
  - Na OOF oszczędność wynosi 30%.
- **Skala ratingowa:** progi klas AAA–CCC na skali 4% to 0,5 / 1 / 2 / 3,5 / 6 / 10%. Obserwowana częstość defaultów rośnie monotonicznie z klasą zarówno na OOF, jak i na teście.

---

## 4. Model black-box — XGBoost

### 4.1 Opis modelu
- **Cechy:**
  - z 170 cech zbioru deweloperskiego usunięto 61 duplikatów (|ρ Spearmana| ≥ 0,95); z każdej pary zostaje cecha o wyższej ważności SHAP, więc model ma 109 cech,
  - `schemat_wsk_bilans` i `schemat_wsk_rzis` są zakodowane one-hot,
  - wartości ±inf zamieniono na NaN, które XGBoost obsługuje natywnie.
- **Strojenie:** `RandomizedSearchCV` (50 iteracji, 4 predefiniowane foldy), wybór najlepszego modelu według PR-AUC.
- **Najlepsze parametry:** `max_depth=4`, `n_estimators=296`, `learning_rate≈0,025`, `min_child_weight=17`, `subsample≈0,89`, `colsample_bytree≈0,75`, `gamma≈1,01`, `reg_alpha≈3,29`, `reg_lambda≈3,44`. Bez ważenia klas.

### 4.2 Wyniki (zbiór testowy, PD surowe)

| Model | Śr. PD | ROC-AUC | PR-AUC | KS | Log-loss | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost (109 cech) | 6,81% | **0,813** | 0,164 | **55,8%** | 0,197 | 0,055 |
| Karta WoE + Platt (8 cech) | 6,78% | 0,790 | **0,173** | 48,5% | 0,197 | **0,053** |

XGBoost lepiej szereguje firmy (wyższe AUC i KS), ale nie poprawia PR-AUC ani funkcji straty, a używa 13 razy więcej cech. Przy progu 4,15% oba modele podejmują tę samą decyzję w 84% przypadków.

### 4.3 Wyjaśnienia globalne
- **SHAP:** ważność cech liczona TreeSHAP (wykres słupkowy i beeswarm na teście).
- **Najważniejsze cechy:** `wsk_index_nowe`, `wsk_index_skr_zaw` i `Kapital_wlasny`, dla których przygotowano wykresy zależności SHAP.
- **PDP, centered ICE i ALE** przygotowano dla `wsk_index_nowe`, `wsk_index_skr_zaw` i `wsk_ogolnego_zadluzenia_1`. ALE stosujemy dlatego, że cechy są silnie skorelowane, a PDP przy korelacjach ekstrapoluje.
- **Porównanie z kartą scoringową:** kształt zależności według SHAP (w log-szansach) dla czterech wspólnych cech jest zgodny z kierunkiem schodków WoE. XGBoost pokazuje jednak dodatkową nieliniowość i rozrzut wynikający z interakcji.

### 4.4 Wyjaśnienia lokalne
- **Metody:** SHAP waterfall i LIME dla 4 firm z testu (niskie ryzyko, firma graniczna, podwyższone ryzyko, wysokie ryzyko).
- **Ocena LIME:** sprawdziliśmy wierność i stabilność (5 uruchomień z różnymi ziarnami losowości).

| Firma (PD) | R² LIME | Jaccard top-5 | Spearman wag |
| :--- | ---: | ---: | ---: |
| 0,4% | 0,17 | 0,63 | 0,36 |
| 6,3% | 0,21 | 0,83 | 0,46 |
| 11,3% | 0,43 | 0,61 | 0,39 |
| 54,0% | 0,59 | 1,00 | 0,52 |

  LIME słabo przybliża model (R² 0,17–0,59), a ranking cech zmienia się między uruchomieniami. **Podstawowym wyjaśnieniem lokalnym jest więc SHAP**, a LIME pełni tylko rolę pomocniczą.
- **Kontrfakt:** dla firmy w klasie BB (PD 7,03%) optymalizacja znalazła wersję z PD 5,19% (klasa BBB) przy praktycznie zerowych zmianach cech (poniżej 0,02 IQR). Firma leży tuż przy granicy podziału drzew. To jest **ryzyko**: wyjaśnienie nie wskazuje realnego działania, które firma mogłaby podjąć, a model można łatwo „ograć” drobnymi zmianami.

### 4.5 Kalibracja i dopasowanie do 4% (zbiór testowy)

| Etap | Śr. PD | Brier | Log-loss | CITL | Nachylenie | Gini |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Surowe | 0,07 | 0,06 | 0,20 | −0,15 | 0,85 | 0,63 |
| Platt | 0,07 | 0,05 | 0,20 | −0,13 | 1,35 | 0,63 |
| Izotoniczna | 0,07 | 0,05 | 0,19 | −0,12 | 1,28 | 0,60 |
| Platt + 4% | 0,04 (4,49%) | 0,05 | 0,20 | 0,32 | 1,35 | 0,63 |

- **Dane do kalibracji:** kalibratory dopasowano na foldzie 4, który zawiera tylko 29 defaultów. Po kalibracji Platta nachylenie wynosi 1,35, czyli model jest teraz **niedostatecznie pewny siebie** (PD zbyt skupione wokół średniej).
- **Przesunięcie do 4%:** δ = −0,450, wyznaczone na foldzie kalibracyjnym. Średnie PD na teście wynosi 4,49%.

### 4.6 Decyzje i ratingi
- **Próg decyzyjny:** wybrany empirycznie na foldzie kalibracyjnym przy kosztach FN = 10 i FP = 1. Wynosi 4,15% na skali 4%.
  - Na teście: TP 30, FP 190, FN 6, TN 373.
  - Akceptacja 63%, koszt na klienta 4,17 wobec 6,01 przy akceptowaniu wszystkich (−31%).
- **Skala ratingowa:** progi to 1 / 2 / 3,5 / 5,5 / 9 / 15%. Na teście monotoniczność jest **złamana** w ogonie: w CCC obserwowany odsetek defaultów wynosi 14,3% przy 21 firmach, a w B 22,6%.

---

## 5. Ograniczenia (oba modele)

- **Mała liczba defaultów:** 180 łącznie, 36 w teście, około 29 na fold. Przedziały ufności metryk są szerokie, a na teście przypada tylko kilka defaultów na klasę ratingową.
- **Brak walidacji out-of-time:** dane nie mają daty sprawozdania, więc nie wiemy, jak modele działają w innym okresie lub cyklu koniunkturalnym.
- **Heurystyczne grupowanie firm:** bez identyfikatora firmy nie można wykluczyć, że rekordy tej samej firmy trafiły do różnych zbiorów.
- **Założony poziom 4%:** to założenie biznesowe, a nie wartość zaobserwowana. W próbie częstość defaultów wynosi 6%.
- **Założone koszty błędów:** stosunek 10:1 jest założeniem, a próg zależy od niego mocno (przy 5:1 karta scoringowa na teście daje stratę względem akceptowania wszystkich).
- **Niespójne konfiguracje między modelami:** każdy model ma własną skalę ratingową i własną metodę wyboru progu (teoretyczny na PD po Platcie w karcie scoringowej, empiryczny na skali 4% w XGBoost), więc ich decyzje nie są w pełni porównywalne.

## 6. Ryzyka

| Ryzyko | Model interpretowalny | Model black-box |
| :--- | :--- | :--- |
| Zgodność regulacyjna i wyjaśnialność | Niskie: wyjaśnienia dokładne, kierunki wpływu cech zgodne z ekonomią | Wysokie: wyjaśnienia post-hoc i przybliżone, LIME niestabilny |
| Przeuczenie | Niskie (8 cech, OOF ≈ test) | Średnie (109 cech, tylko 115 defaultów w treningu) |
| Kalibracja | Dobra od początku | Kalibrator dopasowany na 29 defaultach, nachylenie 1,35 |
| Stabilność | 2 cechy zmieniają znak między foldami (do usunięcia) | Wrażliwość na drobne zmiany przy granicach podziału drzew (kontrfakt) |
| Monotoniczność ratingu | Zachowana na OOF i na teście | Złamana w klasie CCC na teście |
| Manipulacja wynikiem | Ograniczona (skok PD tylko na granicy przedziału WoE) | Wyższa (PD zmienia się przy minimalnych zmianach cech) |
| Fałszywe odrzucenia | Większość odrzuconych firm jest zdrowa (precyzja około 14–16%), więc odrzucenie powinno kierować wniosek do analityka, a nie oznaczać automatycznej odmowy | jw. (FP = 190 na 220 odrzuceń w teście) |

**Rekomendacja:** jako model decyzyjny stosować kartę scoringową WoE, najlepiej w wersji z 5 cechami. XGBoost traktować jako model challenger i benchmark jakości: wskazuje, ile AUC można jeszcze zyskać, i pomaga wykryć nieliniowości do ewentualnego uwzględnienia w karcie.

---

## 7. Plan monitoringu

| Obszar | Miara | Częstotliwość | Próg ostrzegawczy / krytyczny | Działanie |
| :--- | :--- | :--- | :--- | :--- |
| Stabilność populacji | PSI wyniku (PD / punktów) względem zbioru deweloperskiego | miesięcznie | 0,10 / 0,25 | analiza przyczyn / przegląd modelu |
| Stabilność cech | CSI każdej cechy, odsetek braków i wartości ±inf | miesięcznie | CSI 0,10 / 0,25; odsetek braków ponad 2× poziom bazowy | weryfikacja źródła danych |
| Rozkład ratingów | udział firm w klasach AAA–CCC, odsetek akceptacji | miesięcznie | zmiana udziału klasy > 5 p.p. | przegląd progów |
| Jakość rankingu | ROC-AUC / Gini, KS na dojrzałych obserwacjach (12 mies.) | kwartalnie / rocznie | AUC < 0,75 / < 0,70 | rekalibracja / ponowne zbudowanie modelu |
| Kalibracja ogólna | średnie PD wobec obserwowanej częstości defaultów; CITL | kwartalnie | odchylenie > 20% względne / CITL poza [−0,3; 0,3] | aktualizacja przesunięcia δ lub kalibratora |
| Kalibracja w klasach | test dwumianowy / Jeffreysa dla każdej klasy; monotoniczność częstości defaultów | rocznie | więcej niż 1 klasa odrzucona na poziomie 5% lub odwrócenie kolejności klas | przegląd skali ratingowej |
| Opłacalność decyzji | koszt na firmę i oszczędność względem akceptowania wszystkich; odsetek ręcznych zmian decyzji (override) | kwartalnie | oszczędność < 10% / override > 15% | przegląd progu i założeń kosztowych |
| Stabilność wyjaśnień (black-box) | ranking średnich \|SHAP\| (korelacja Spearmana z rankingiem bazowym) | kwartalnie | ρ < 0,8 | analiza dryfu, przegląd modelu |
| Stabilność współczynników (karta) | β po ponownym dopasowaniu na nowych danych: znak i wartość | rocznie | zmiana znaku lub \|Δβ\| > 2 odch. std. z CV | przegląd selekcji cech |

**Zasady ogólne:**
- **Dojrzewanie obserwacji:** default ocenia się po 12 miesiącach, więc metryki jakości liczymy tylko na obserwacjach dojrzałych. Do tego czasu monitorujemy tylko stabilność populacji i cech.
- **Przegląd pełny:** co najmniej raz w roku oraz zawsze po przekroczeniu progu krytycznego. Obejmuje ponowną walidację na nowych danych (out-of-time), weryfikację założenia 4% i kosztów błędów.
- **Logowanie:** każda decyzja zapisywana jest z wersją modelu, PD, ratingiem i wyjaśnieniem (punkty karty lub top SHAP), żeby można ją było odtworzyć na potrzeby audytu i reklamacji.
