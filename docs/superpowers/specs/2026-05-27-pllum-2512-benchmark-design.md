# PLLuM 2512 — benchmark redempcji + dual-metric (format vs matematyka)

**Data:** 2026-05-27
**Autor:** Greg Brzezinka (Prosit AS)
**Status:** zatwierdzony design, do przejścia w plan implementacji

## Kontekst

W benchmarku egzaminu ósmoklasisty z matematyki (arkusz CKE 12 maja 2026, 20 zadań,
30 pkt) testowaliśmy generację PLLuM **2412** (grudzień 2024). Oba warianty —
`Llama-PLLuM-8B-instruct` i natywny `PLLuM-12B-instruct` — dostały **3/30 (10%)**.
Opublikowany post na LinkedIn opisał to jako „PLLuM nie skaluje się z rozmiarem".

Analiza surowych odpowiedzi + pomiar na danych (2026-05-27) **obaliły** wstępną
hipotezę „to artefakt parsowania". Hojny, format-agnostyczny parser odzyskuje
w całym benchmarku **dokładnie 1 punkt** (PLLuM-12B zad. 5: „D. 0,6x+0,8y"); zadania
otwarte sędzia i tak ocenia z pełnego `raw`. Skorygowane wyniki: Llama-PLLuM 8B
**3/30 → 3/30** (bez zmian), PLLuM 12B **3/30 → 4/30** (+1). **3/30 jest realne, to nie
bug harnessa.**

Co naprawdę się dzieje (i to jest właściwy insight):
- PLLuM **rozumuje** — pełne rozwiązania krok po kroku (stare „bez rozumowania" było
  błędne; w nowym poście tego nie powtarzamy).
- **Ignoruje strukturę zadań zamkniętych** — nie wybiera z A/B/C/D, liczy free-form
  w stylu GSM8K (`#### 460`), często bez litery do sparsowania.
- **Myli rachunki**: zad. 2 → 9944 zamiast 4994; zad. 7 → 460 jabłek zamiast 230.

Bielik i Gemma trzymały format `<odpowiedz>` i mają realne wyniki (re-score parserem
hojnym nic im nie zmienia — zweryfikowane na wszystkich 8 modelach).

Na HF jest nowsza generacja — sufiks **2512** (pliki ruszane 20 maja 2026), w tym
zupełnie nowy rozmiar **4B**, który nie istniał w 2412.

## Cel

1. Uruchomić generację PLLuM 2512 na tym samym arkuszu (apples-to-apples).
2. Wprowadzić **dwie metryki** odróżniające zdolność matematyczną od trzymania
   formatu wyjścia — i przeliczyć nimi wszystkie modele.
3. Wydać lekki follow-up na LinkedIn: „PLLuM nie trzyma formatu — czy nowa wersja
   trzyma?".

## Zakres modeli (decyzja: pełna para + nowy 4B)

| Model HF | Rola |
|---|---|
| `CYFRAGOVPL/Llama-PLLuM-8B-instruct-2512` | następca 8B z 3/30 |
| `CYFRAGOVPL/PLLuM-12B-instruct-2512` | następca 12B z 3/30 (apples-to-apples) |
| `CYFRAGOVPL/PLLuM-4B-instruct-2512` | nowy rozmiar; klasa Bielika 4.5B i Gemmy 4 E4B |

Każdy: lokalna konwersja MLX **Q8** (jak przy 2412), **identyczny** prompt
(`prompts/system_*.txt`), temperatura 0, te same neutralne opisy rysunków.

## Metodologia: hojny parser + jedna nota (decyzja: bez kolumny dual-metric)

Delta artefaktu = 1 pkt, więc osobna kolumna „format vs matma" nie ma sensu. Zamiast
tego: jeden, celowo **hojny** parser zamkniętych w `05_ocen.py` + krótka nota
metodyczna.

### Rozszerzony `wylusk_zamkniete`
Do istniejących fallbacków (`<odpowiedz>`, `\boxed{}`, „Odpowiedź X", litera na końcu)
dodać, gdy nic nie znaleziono:
1. `####\s*([A-DPF]{1,2})` (GSM8K z literą)
2. `(?:odpowiedź to|wynik|final)…([A-DPF]{1,2})`
3. ostatnia linia typu `^\s*([A-DPF]{1,2})[.):]` (łapie „D. 0,6x+0,8y")

**Zwalidowane** na wszystkich 8 modelach (2026-05-27): parser monotoniczny — żaden
model nie traci punktów; jedyny realny zysk to PLLuM-12B zad. 5. Bielik/Gemma bez zmian.

### Re-score z `raw`
Scorer re-parsuje z `raw`, więc historyczne runy dostają fair ocenę bez ponownej
inferencji. Zadania otwarte: sędzia Claude **już** dostaje pełne `raw`
(`05_ocen.py:88`), więc te wyniki są realne i bez zmian.

### Nota metodyczna (do README/raportu)
„Parser zamkniętych celowo hojny (łapie też format GSM8K `####` i odpowiedzi opisowe).
PLLuM i tak osiąga 3–4/30 — niski wynik nie jest artefaktem harnessa."

## Deliverables

1. **Skrypty uruchomieniowe** dla 3 modeli 2512 (wzorzec: `04d_run_pllum12.py`).
2. **`results/raport.md`** + **tabela w README**: nowa kolumna (compliance formatu vs
   wynik realny) + krótka notka metodyczna o dual-metric. Ranking może się zmienić —
   zaakceptowane.
3. **Follow-up LinkedIn** w `temp/linkedin_post_pllum.md`: polski, główny post + wątek
   komentarzy, styl i limity znaków jak w poprzednim `temp/linkedin_post.md`.

## Narracja posta (lekka, do przodu — bez korekty starego posta)

- **Hook:** „Pytaliście, jak PLLuM poradzi sobie z egzaminem. Już go testowałem —
  i jest ciekawiej, niż się wydaje."
- **Sedno failure mode:** PLLuM *rozumuje* (pełne rozwiązania), ale (a) **ignoruje
  strukturę zadań zamkniętych** — nie wybiera z A/B/C/D, liczy free-form w stylu GSM8K,
  (b) **myli rachunki**. Stąd 3/30, nie z braku „myślenia".
- **Dowód rzetelności:** przebudowałem parser na maksymalnie hojny — PLLuM i tak ~3–4/30.
  To nie wina harnessa.
- **Twist:** nowa generacja 2512 (+ nowy 4B) — zachowuje się lepiej?
  *(payload zależny od runu; ton: „a może? zobaczymy")*
- **Lekcja:** „polski model" ≠ dobry model; rozmiar bez właściwego treningu nie kupuje
  wyniku; a sposób, w jaki model *przegrywa* (ignorowanie formatu MCQ), bywa
  ciekawszy niż sam wynik.

**Świadomie poza zakresem:** nie edytujemy opublikowanego posta, nie piszemy erraty,
nie przepraszamy. Nowy post jest prawdziwy sam w sobie i po prostu nie powtarza
nieaktualnego „bez rozumowania".

## Ryzyka i decyzje

- **4B bez tagu architektury** — `mlx-lm convert` może nie obsłużyć. Plan B:
  `featherless-ai` (12B ma live inference) albo pominięcie 4B z jawną notką w raporcie.
- **Re-score zmienia opublikowane liczby w README** — zaakceptowane (b).
- **Spójność z opublikowanym postem** — świadoma decyzja właściciela (a): stary post
  zostaje; nowy nie powiela fałszywej tezy.

## Out of scope

- Konwersja/publikacja wag MLX dla PLLuM (tylko lokalna konwersja na potrzeby runu).
- Warianty `chat`/`base`/`nc` oraz 70B.
- Zmiany w pipelinie Norway (`norway/`).
