# Instrukcja Uruchomienia BrainAccess (LSL)

Aby połączyć opaskę z naszym skryptem, musisz włączyć strumieniowanie LSL w przeglądarce.

## Krok 1: Konfiguracja w Przeglądarce (BrainAccess)
1.  Upewnij się, że opaska jest połączona (Status: **Connected**).
2.  Przejdź do zakładki **Stream** (lub **LSL**).
3.  Znajdź przełącznik **LSL Stream** lub przycisk **Enable LSL**.
4.  **Włącz go**.
    *   *Ważne:* Jeśli widzisz pole "Stream Name", zostaw domyślne lub upewnij się, że typ to `EEG`.

## Krok 2: Uruchomienie Skryptu
Wróć do terminala i wpisz:

```bash
python3 brainaccess_live.py
```

## Co powinno się stać?
1.  Skrypt wyświetli: `Szukanie strumienia EEG (LSL)...`
2.  Gdy wykryje opaskę: `Połączono z BrainAccess!`
3.  Zacznie wypisywać emocje: `Stan: POSITIVE | Pewność: 0.95`

Jeśli skrypt "wisi" na szukaniu, sprawdź czy w przeglądarce LSL jest na pewno włączone (czasem trzeba odświeżyć stronę).
