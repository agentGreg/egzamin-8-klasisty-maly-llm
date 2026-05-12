# Raport — Egzamin Ósmoklasisty z Matematyki, 12 maja 2026

Benchmark trzech konfiguracji modeli ~4-5B parametrów uruchamianych lokalnie przez MLX na Apple M5 Max.

## Wyniki

| Model                             | Wynik       | Procent   | Zamknięte   | Otwarte   |
|-----------------------------------|-------------|-----------|-------------|-----------|
| Bielik 4.5B v3 (8-bit, text-only) | **24 / 30** | **80.0%** | 12/14       | 12/16 pkt |
| Gemma 3 4B IT (4-bit, text-only)  | **18 / 30** | **60.0%** | 9/14        | 9/16 pkt  |
| Gemma 3 4B IT (4-bit, multimodal) | **14 / 30** | **46.7%** | 8/14        | 6/16 pkt  |

(Próg zdawalności egzaminu nie jest formalnie ustanowiony — wynik to liczba zdobytych punktów na 30 możliwych.)

## Tabela szczegółowa

| zad   | typ   |   max | klucz                      | Bielik txt odp   | Bielik txt pkt   | Gemma txt odp          | Gemma txt pkt   | Gemma MM odp           | Gemma MM pkt   |
|-------|-------|-------|----------------------------|------------------|------------------|------------------------|-----------------|------------------------|----------------|
| z01   | zamk  |     1 | A                          | A                | 1/1              | A                      | 1/1             | A                      | 1/1            |
| z02   | zamk  |     1 | B                          | B                | 1/1              | D                      | 0/1             | A                      | 0/1            |
| z03   | zamk  |     1 | C                          | C                | 1/1              | C                      | 1/1             | C                      | 1/1            |
| z04   | zamk  |     1 | A                          | A                | 1/1              | A                      | 1/1             | A                      | 1/1            |
| z05   | zamk  |     1 | D                          | D                | 1/1              | C                      | 0/1             | C                      | 0/1            |
| z06   | zamk  |     1 | AC                         | BD               | 0/1              | D                      | 0/1             | BD                     | 0/1            |
| z07   | zamk  |     1 | B                          | B                | 1/1              | B                      | 1/1             | B                      | 1/1            |
| z08   | zamk  |     1 | BD                         | BD               | 1/1              | BD                     | 1/1             | BD                     | 1/1            |
| z09   | zamk  |     1 | D                          | D                | 1/1              | D                      | 1/1             | C                      | 0/1            |
| z10   | zamk  |     1 | PP                         | PP               | 1/1              | P1                     | 0/1             | P1                     | 0/1            |
| z11   | zamk  |     1 | C                          | C                | 1/1              | C                      | 1/1             | C                      | 1/1            |
| z12   | zamk  |     1 | PP                         | PP               | 1/1              | P1                     | 0/1             | PF                     | 0/1            |
| z13   | zamk  |     1 | D                          | B                | 0/1              | D                      | 1/1             | D                      | 1/1            |
| z14   | zamk  |     1 | A                          | A                | 1/1              | A                      | 1/1             | A                      | 1/1            |
| z15   | otw   |     2 | Ela przygotowała 57 karte… | —                | 2/2              | Ela przygotowała 57 ka | 2/2             | Ela przygotowała 57 ka | 1/2            |
| z16   | otw   |     3 | Przejazd z Jodłowa do Dęb… | —                | 3/3              | —                      | 2/3             | —                      | 3/3            |
| z17   | otw   |     3 | Liczba dzieci na turnieju… | —                | 2/3              | Procent liczby wszystk | 1/3             | Liczba procentów wszys | 0/3            |
| z18   | otw   |     2 | Objętość ostrosłupa ACDS … | —                | 2/2              | Objętość ostrosłupa AC | 0/2             | Objętość ostrosłupa AC | 0/2            |
| z19   | otw   |     3 | Pani Anna zapłaci 142,80 … | —                | 3/3              | Pani Anna musi zapłaci | 3/3             | Pani Anna musi zapłaci | 2/3            |
| z20   | otw   |     3 | Obwód równoległoboku KLMN… | —                | 0/3              | Obwód równoległoboku K | 1/3             | —                      | 0/3            |

## Wydajność

| Model                             | Łączny czas   | Średni / zadanie   | Throughput   |
|-----------------------------------|---------------|--------------------|--------------|
| Bielik 4.5B v3 (8-bit, text-only) | 113 s         | 5.7 s              | 81 tok/s     |
| Gemma 3 4B IT (4-bit, text-only)  | 65 s          | 3.3 s              | 157 tok/s    |
| Gemma 3 4B IT (4-bit, multimodal) | 68 s          | 3.4 s              | 151 tok/s    |

## Metodyka

- **Arkusz**: oficjalny PDF CKE z 12 maja 2026, 20 zadań (1–14 zamknięte ABCD/PF, 15–20 otwarte), max 30 pkt.
- **Klucz odpowiedzi**: wygenerowany przez Claude Opus 4.7 z PDF jako kontekst, następnie **ręcznie zweryfikowany** (Claude pomylił się w 5 zadaniach — głównie copy-paste między rozumowaniem a polem `odpowiedz`).
- **Runtime**: `mlx-lm` / `mlx-vlm` na Apple M5 Max, 128 GB unified memory.
- **Gemma multimodal**: przez `mlx-vlm` — model widzi obrazki zadań bezpośrednio.
- **Gemma text-only**: ten sam model przez `mlx-lm`, dostaje tekstowe opisy rysunków zamiast obrazków (uczciwy compare z Bielikiem).
- **Bielik 4.5B v3** (text-only): dostaje te same opisy rysunków.
- **Ocena zadań otwartych**: Claude Opus 4.7 wg kryteriów CKE (pełna metoda + wynik, błąd rachunkowy, brak postępu).
- **Temperatura**: 0 dla wszystkich modeli (deterministyczne odpowiedzi).
- **Parser odpowiedzi**: preferuje `<odpowiedz>` → `\boxed{}` → ostatnie „Odpowiedź: X" → fallback.
