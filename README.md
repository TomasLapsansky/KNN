# KNN PROJEKT - REIDENTIFIKÁCIA VOZIDIEL
## Projekt do KNN

### Abstrakt
Tento repozitár obsahuje riešenie problému re-identifikácie áut zo snímkov zozbieraných z rôznych kamier. V práci sa predpokladá že samotné vyhľadávanie aut v obraze je už vyriešenou úlohou a zameriava sa na re-identifikáciu vozidiel z rôznych kamier. V posledných rokoch sa re-identifikácia áut stáva  dôležitou súčasťou pri vytváraní inteligentných systémov v doprave. Ako príklad môžme uviesť identifikáciu aut na spoplatnených úsekoch alebo pri mýtnych systémoch. Táto práca sa zameriava na re-identifikáciu vozidiel na základe jeho vlastnosti (tvar, farba, typ vozidla, atď) bez toho aby bolo rozpoznávané evidenčné číslo vozidla.


### Spustenie projektu
Spustenie projektu je možné nasledovne:
```sh
python3 model.py [-c|-t] <názov checkpointu>
```
* **-c** -- spustenie trenovania od existujúceho checkpointu
* **-t** -- spustenie evaluacie nad dátami

V pripade nezadania parametrov sa spúšťa trenovanie siete od začiatku.



