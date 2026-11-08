#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robot de trading FICTIF Bitcoin/EUR, 24 h/24 sur Mac.

Aucun ordre reel. Aucune cle API. Aucun compte necessaire.
Les 4 portefeuilles sont independants; chacun part de 500 EUR fictifs.
Python 3.10+ ; pip install requests scikit-learn

Commandes :
  python3 robot_papier_24_7.py --test       # Tests locaux sans Internet
  python3 robot_papier_24_7.py --backtest   # Evaluation historique recente
  python3 robot_papier_24_7.py --once       # Une lecture des cours, puis sortie
  caffeinate -dims python3 robot_papier_24_7.py  # 24h/24, jusqu'a Ctrl+C
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Installe requests : python3 -m pip install requests")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
except ImportError:
    sys.exit("Installe l'IA : python3 -m pip install scikit-learn")

BASE = Path(__file__).resolve().parent
ETAT = BASE / "etat_robot_24_7.json"
JOURNAL = BASE / "journal_robot_24_7.csv"
RAPPORT = BASE / "bilan_robot_24_7.txt"
VERROU = BASE / ".robot_24_7.lock"
HISTORIQUE = BASE / "historique_bougies_15min.csv"
API = "https://api.kraken.com/0/public"
PAIRE = "XBTEUR"                    # Bitcoin / euro, marche spot public Kraken
INTERVALLE_MIN = 15               # Une decision toutes les 15 min
POLL_SECONDES = 60                # Releve un cours par minute
CAPITAL_INITIAL = 500.0           # Par portefeuille, FICTIF
FRAIS = 0.001                     # Hypothese: 0,1% par operation
GLISSEMENT = 0.0005               # Hypothese: 0,05% par operation
EXPOSITION_MAX = 0.30             # Au plus 30% de la valeur, a l'achat
PAUSE_BARRES = 8                  # Minimum 2 heures entre transactions
STOP_PERTE_PORTF = 6.0           # Bloque et vend fictivement si baisse depuis le pic >= 6%
NOMS = ("ia", "tendance", "cassure", "conservation_30", "conservation_100")
CHAMPS_JOURNAL = (
    "utc", "bougie_utc", "strategie", "signal", "action", "prix_eur",
    "euros", "bitcoin", "valeur_eur", "valeur_liquidation_eur",
    "frais_cumules_eur", "achats", "ventes", "perte_max_pct", "bloque"
)


def horodatage(ts: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if ts is None else ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fichier_atomique(path: Path, content: str) -> None:
    nouveau = path.with_suffix(path.suffix + ".tmp")
    with nouveau.open("w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(nouveau, path)


@dataclass(frozen=True)
class Bougie:
    ts: int
    ouverture: float
    plus_haut: float
    plus_bas: float
    cloture: float


def lire_api(endpoint: str, params: dict) -> dict:
    r = requests.get(
        f"{API}/{endpoint}", params=params, timeout=20,
        headers={"User-Agent": "RobotPapier24h-Education/1.0"},
    )
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise RuntimeError("Kraken : " + "; ".join(data["error"]))
    if not data.get("result"):
        raise RuntimeError("Kraken : reponse vide")
    return data["result"]


def lire_bougies() -> list[Bougie]:
    data = lire_api("OHLC", {"pair": PAIRE, "interval": INTERVALLE_MIN})
    key = next(k for k in data if k != "last")
    # La toute derniere bougie Kraken est EN COURS : on l'ignore.
    source = data[key][:-1]
    bougies = [
        Bougie(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))
        for x in source
    ]
    bougies = sorted({b.ts: b for b in bougies}.values(), key=lambda b: b.ts)
    if len(bougies) < 150:
        raise RuntimeError("Historique de 15 min insuffisant (moins de 150 bougies)")
    if any(not all(math.isfinite(z) and z > 0 for z in (b.ouverture, b.plus_haut, b.plus_bas, b.cloture))
           for b in bougies):
        raise RuntimeError("Prix historique invalide")
    return bougies


def lire_ticker() -> float:
    data = lire_api("Ticker", {"pair": PAIRE})
    key = next(iter(data))
    cours = float(data[key]["c"][0])
    if not math.isfinite(cours) or cours <= 0:
        raise RuntimeError("Ticker invalide")
    return cours


def ema(valeurs: list[float], n: int) -> float:
    alpha = 2 / (n + 1)
    moyenne = valeurs[0]
    for x in valeurs[1:]:
        moyenne += alpha * (x - moyenne)
    return moyenne


def caracteristiques(bougies: list[Bougie], i: int) -> list[float]:
    """Features a la cloture i. Aucun recours a i+1 (pas de futur)."""
    if i < 64:
        raise ValueError("Pas assez de bougies")
    c = [b.cloture for b in bougies[:i+1]]
    rendements = [c[k] / c[k-1] - 1 for k in range(i-15, i+1)]
    vol = statistics.pstdev(rendements)
    gains = sum(max(0.0, c[k] - c[k-1]) for k in range(i-13, i+1))
    pertes = sum(max(0.0, c[k-1] - c[k]) for k in range(i-13, i+1))
    rsi = gains / (gains + pertes) if gains + pertes > 0 else .5
    return [
        c[i] / c[i-1] - 1,
        c[i] / c[i-4] - 1,
        c[i] / c[i-16] - 1,
        ema(c[-60:], 12) / ema(c[-60:], 48) - 1,
        vol,
        rsi,
    ]


def construire_ia(bougies: list[Bougie], indice_fin_entrainement: int | None = None):
    """Entraine sans connaitre l'avenir; validation chronologique informative."""
    n = len(bougies)
    if n < 180:
        raise RuntimeError("Pas assez de prix pour l'IA")
    x, y, indices = [], [], []
    for i in range(64, n-1):
        if indice_fin_entrainement is not None and i + 1 >= indice_fin_entrainement:
            continue
        # Ne pas inventer des rendements si le Mac etait arrete pendant des heures.
        if any(bougies[k].ts - bougies[k-1].ts != INTERVALLE_MIN * 60
               for k in range(i-63, i+2)):
            continue
        x.append(caracteristiques(bougies, i))
        y.append(int(bougies[i+1].cloture > bougies[i].cloture))
        indices.append(i)
    if len(set(y)) < 2 or len(x) < 50:
        raise RuntimeError("Donnees d'entrainement insuffisamment variees")

    # Pour la version deployee, score d'un modele distinct sur le dernier 30%.
    validation = "non calculee"
    if indice_fin_entrainement is None:
        sep = int(len(x) * .70)
        if len(set(y[:sep])) == 2 and sep > 60 and len(y) - sep > 20:
            modele_test = RandomForestClassifier(n_estimators=100, max_depth=4,
                            min_samples_leaf=10, random_state=42, n_jobs=1)
            modele_test.fit(x[:sep], y[:sep])
            prediction = modele_test.predict(x[sep:])
            precision = accuracy_score(y[sep:], prediction)
            reference = max(sum(y[sep:]), len(y[sep:]) - sum(y[sep:])) / len(y[sep:])
            validation = f"precision {precision:.1%}, reference majoritaire {reference:.1%} (historique court)"

    modele = RandomForestClassifier(n_estimators=100, max_depth=4,
                                   min_samples_leaf=10, random_state=42, n_jobs=1)
    modele.fit(x, y)
    return modele, validation


def proba_hausse(modele, bougies: list[Bougie]) -> float:
    x = caracteristiques(bougies, len(bougies)-1)
    p = modele.predict_proba([x])[0]
    return float(dict(zip(modele.classes_, p)).get(1, 0.0))


def signaux(bougies: list[Bougie], modele) -> dict[str, str]:
    c = [b.cloture for b in bougies]
    if len(c) < 80:
        return {n: "ATTENDRE" for n in NOMS}
    fast = ema(c[-65:], 12)
    slow = ema(c[-65:], 48)
    latest = c[-1]
    p = proba_hausse(modele, bougies) if modele is not None else .5
    ia = "ACHAT" if p >= .65 and fast > slow else "VENTE" if p <= .40 or fast < slow else "ATTENDRE"
    tendance = "ACHAT" if fast > slow * 1.002 and latest > slow else "VENTE" if fast < slow or latest < slow * .997 else "ATTENDRE"
    # Exclure la bougie actuelle des niveaux de cassure.
    haut_precedent = max(b.plus_haut for b in bougies[-33:-1])
    bas_precedent = min(b.plus_bas for b in bougies[-17:-1])
    cassure = "ACHAT" if latest > haut_precedent and fast > slow else "VENTE" if latest < bas_precedent or latest < slow else "ATTENDRE"
    return {"ia": ia, "tendance": tendance, "cassure": cassure,
            "conservation_30": "GARDER", "conservation_100": "GARDER"}


@dataclass
class Portefeuille:
    euros: float = CAPITAL_INITIAL
    bitcoin: float = 0.0
    frais_cumules: float = 0.0
    achats: int = 0
    ventes: int = 0
    pic: float = CAPITAL_INITIAL
    perte_max: float = 0.0
    derniere_bougie_trajet: int = -10**10  # timestamp UNIX de la derniere transaction fictive
    bloque: bool = False

    def valeur(self, cours: float) -> float:
        return self.euros + self.bitcoin * cours

    def valeur_liquidation(self, cours: float) -> float:
        # Estime ce qui resterait si la position etait vendue MAINTENANT.
        if self.bitcoin <= 0:
            return self.euros
        return self.euros + self.bitcoin * cours * (1 - GLISSEMENT) * (1 - FRAIS)

    def acheter(self, cours: float, bar: int, part: float = EXPOSITION_MAX) -> bool:
        if self.bitcoin > 1e-10 or self.bloque:
            return False
        montant = min(self.euros, self.valeur(cours) * part)
        if montant < 5:
            return False
        self.frais_cumules += montant * FRAIS
        self.euros -= montant
        self.bitcoin += montant * (1-FRAIS) / (cours*(1+GLISSEMENT))
        self.achats += 1
        self.derniere_bougie_trajet = bar
        return True

    def vendre(self, cours: float, bar: int) -> bool:
        if self.bitcoin <= 0:
            return False
        brut = self.bitcoin * cours * (1-GLISSEMENT)
        self.frais_cumules += brut * FRAIS
        self.euros += brut * (1-FRAIS)
        self.bitcoin = 0.0
        self.ventes += 1
        self.derniere_bougie_trajet = bar
        return True

    def mesurer(self, cours: float) -> None:
        valeur = self.valeur_liquidation(cours)
        self.pic = max(self.pic, valeur)
        self.perte_max = max(self.perte_max, (self.pic-valeur)/self.pic * 100)

    def risque(self, cours: float, bar: int, est_benchmark: bool = False) -> bool:
        self.mesurer(cours)
        if est_benchmark or self.bloque:
            return False
        if (self.pic - self.valeur_liquidation(cours)) / self.pic * 100 >= STOP_PERTE_PORTF:
            self.vendre(cours, bar)
            self.bloque = True
            self.mesurer(cours)
            return True
        return False

    def appliquer(self, signal: str, cours: float, bar: int) -> str:
        if self.bloque:
            return "BLOQUEE"
        if bar-self.derniere_bougie_trajet < PAUSE_BARRES * INTERVALLE_MIN * 60:
            return "ATTENTE"
        if signal == "ACHAT" and self.acheter(cours, bar):
            return "ACHAT FICTIF"
        if signal == "VENTE" and self.vendre(cours, bar):
            return "VENTE FICTIVE"
        return "RIEN"


def creation_etat(cours: float, bar: int) -> dict:
    portfolios = {name: Portefeuille() for name in NOMS}
    portfolios["conservation_30"].acheter(cours, bar, part=.30)
    portfolios["conservation_100"].acheter(cours, bar, part=1.0)
    return {"version": 1, "debut": horodatage(), "last_bar": -1,
            "last_model_time": 0, "modele_validation": "non calculee", "erreurs": 0,
            "portefeuilles": {name: asdict(p) for name, p in portfolios.items()}}


def etat_charger() -> dict | None:
    if not ETAT.exists():
        return None
    with ETAT.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != 1 or set(data["portefeuilles"]) != set(NOMS):
        raise RuntimeError("Etat incompatible. Ne le supprime pas; fais une sauvegarde avant toute modification.")
    return data


def etat_sauver(etat: dict) -> None:
    fichier_atomique(ETAT, json.dumps(etat, indent=2, ensure_ascii=False))


def extraire(etat: dict) -> dict[str, Portefeuille]:
    return {name: Portefeuille(**etat["portefeuilles"][name]) for name in NOMS}


def placer(etat: dict, pf: dict[str, Portefeuille]) -> None:
    etat["portefeuilles"] = {name: asdict(x) for name, x in pf.items()}


def journaliser(nom: str, signal: str, action: str, pf: Portefeuille,
                 cours: float, ts_bougie: int) -> None:
    nouveau = not JOURNAL.exists() or JOURNAL.stat().st_size == 0
    ligne = {"utc": horodatage(), "bougie_utc": horodatage(ts_bougie),
             "strategie": nom, "signal": signal, "action": action,
             "prix_eur": round(cours, 2), "euros": round(pf.euros, 4),
             "bitcoin": round(pf.bitcoin, 10), "valeur_eur": round(pf.valeur(cours), 4),
             "valeur_liquidation_eur": round(pf.valeur_liquidation(cours), 4),
             "frais_cumules_eur": round(pf.frais_cumules, 4),
             "achats": pf.achats, "ventes": pf.ventes,
             "perte_max_pct": round(pf.perte_max, 3), "bloque": pf.bloque}
    with JOURNAL.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CHAMPS_JOURNAL)
        if nouveau:
            writer.writeheader()
        writer.writerow(ligne)


def conserver_bougies(bougies: list[Bougie]) -> None:
    """Archive les bougies terminees au fil du temps, sans doublons."""
    anciennes = set()
    if HISTORIQUE.exists():
        with HISTORIQUE.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                anciennes.add(int(row["ts"]))
    ajouts = [b for b in bougies if b.ts not in anciennes]
    if ajouts:
        nouveau = not HISTORIQUE.exists()
        with HISTORIQUE.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if nouveau:
                writer.writerow(["ts", "utc", "open", "high", "low", "close"])
            for b in ajouts:
                writer.writerow([b.ts, horodatage(b.ts), b.ouverture,
                                 b.plus_haut, b.plus_bas, b.cloture])


def charger_bougies_archive(max_points: int = 3000) -> list[Bougie]:
    """Utilise aussi l'historique collecte sur le Mac au fil des semaines.

    3000 bougies ~= 31 jours. Les interruptions du Mac ne sont pas inventees.
    """
    if not HISTORIQUE.exists():
        return []
    with HISTORIQUE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bars = [Bougie(int(r["ts"]), float(r["open"]), float(r["high"]),
                   float(r["low"]), float(r["close"])) for r in rows]
    return sorted({b.ts: b for b in bars}.values(), key=lambda b: b.ts)[-max_points:]


def rapport(etat: dict, pf: dict[str, Portefeuille], cours: float) -> str:
    lignes = ["ROBOT BTC/EUR - SIMULATION 24 H / 24", "AUCUN ORDRE REEL",
              f"Debut: {etat['debut']}", f"Derniere mise a jour: {horodatage()}",
              f"Dernier prix: {cours:,.2f} EUR", "Capital initial : 500 EUR FICTIFS PAR STRATEGIE",
              f"Frais: {FRAIS:.2%} / cote ; glissement: {GLISSEMENT:.2%} / cote",
              "Evaluation sur prix de liquidation (frais de sortie inclus).",
              f"IA : {etat.get('modele_validation', 'non calculee')}", ""]
    scores = []
    for name, p in pf.items():
        liquidation = p.valeur_liquidation(cours)
        scores.append((liquidation, name))
        lignes.append(f"{name:18} {liquidation:8.2f} EUR  "
                      f"P/L {liquidation-500:+8.2f} EUR  "
                      f"achats {p.achats:3} ventes {p.ventes:3}  "
                      f"frais {p.frais_cumules:6.2f}  drawdown {p.perte_max:5.2f}% "
                      f"{'RISQUE STOP' if p.bloque else ''}")
    scores.sort(reverse=True)
    lignes += ["", "Le meilleur score affiche n'est PAS une preuve de rentabilite future.",
               "Ne pas choisir une methode sur une seule semaine ou quelques transactions.",
               f"Journal des decisions : {JOURNAL.name}",
               f"Bougies archivees : {HISTORIQUE.name}"]
    return "\n".join(lignes) + "\n"


def observer_une_fois(etat: dict | None, modele=None):
    bougies = lire_bougies()
    cours = lire_ticker()
    derniere = bougies[-1]
    conserver_bougies(bougies)
    if etat is None:
        etat = creation_etat(cours, derniere.ts)
        pf = extraire(etat)
        for name in ("conservation_30", "conservation_100"):
            pf[name].mesurer(cours)
            journaliser(name, "REFERENCE", "ACHAT INITIAL FICTIF", pf[name], cours, derniere.ts)
        placer(etat, pf)
        etat_sauver(etat)
        print("NOUVELLE SIMULATION : 500 EUR fictifs par methode.")

    # Train on closed candles only; retrain at most once per 24 hours.
    if modele is None or time.time() - etat.get("last_model_time", 0) >= 24*3600:
        archive = charger_bougies_archive()
        modele, validation = construire_ia(archive if len(archive) >= len(bougies) else bougies)
        etat["last_model_time"] = time.time()
        etat["modele_validation"] = validation
        print("MODELE IA:", validation)

    pf = extraire(etat)
    # Un signal uniquement sur les bougies TERMINEES, jamais sur la bougie en cours.
    # Si flux trop vieux, ne pas simuler une transaction.
    age = time.time() - (derniere.ts + INTERVALLE_MIN*60)
    if age > 30 * 60:
        raise RuntimeError("Derniere bougie trop ancienne, aucun ordre fictif")

    # Surveille l'exposition toutes les 60 s, meme sans nouvelle bougie.
    for name, p in pf.items():
        if p.risque(cours, derniere.ts, name.startswith("conservation")):
            journaliser(name, "STOP", "VENTE FICTIVE + BLOCAGE", p, cours, derniere.ts)
            print(f"ALERTE RISQUE - {name}: position fermee, nouveaux achats bloques")

    if derniere.ts > etat["last_bar"]:
        etat["last_bar"] = derniere.ts
        signaux_now = signaux(bougies, modele)
        for name, p in pf.items():
            signal = signaux_now[name]
            action = p.appliquer(signal, cours, derniere.ts) if name in ("ia", "tendance", "cassure") else "CONSERVE"
            p.mesurer(cours)
            journaliser(name, signal, action, p, cours, derniere.ts)
            print(f"{name:18} {signal:9} {action:15} "
                  f"capital liquidable {p.valeur_liquidation(cours):8.2f} EUR")
        print(f"{horodatage()} | BTC/EUR {cours:,.2f} EUR | "
              f"Bougie close {horodatage(derniere.ts)}\n")

    placer(etat, pf)
    etat_sauver(etat)
    fichier_atomique(RAPPORT, rapport(etat, pf, cours))
    return etat, modele


def backtest_recentre(bougies: list[Bougie]) -> str:
    """Test historique reserve: regles fixes, frais et execution au prochain OPEN."""
    n = len(bougies)
    sep = int(n*.70)
    modele, _ = construire_ia(bougies[:sep])
    pfs = {name: Portefeuille() for name in NOMS}
    for name, proportion in (("conservation_30", .30), ("conservation_100", 1.)):
        pfs[name].acheter(bougies[sep].ouverture, bougies[sep].ts, proportion)
    for i in range(sep-1, n-1):
        # Signal a partir de la bougie i, transaction fictive a l'OPEN de i+1.
        sub = bougies[:i+1]
        sign = signaux(sub, modele)
        prix_exec = bougies[i+1].ouverture
        for name, p in pfs.items():
            p.risque(prix_exec, bougies[i+1].ts, name.startswith("conservation"))
            if name not in ("conservation_30", "conservation_100"):
                p.appliquer(sign[name], prix_exec, bougies[i+1].ts)
            p.mesurer(bougies[i+1].cloture)
    fin = bougies[-1].cloture
    output = [f"BACKTEST EXPERIMENTAL: {n} bougies 15 min, derniers 30% reserves au test.",
              "IA entrainee UNIQUEMENT sur le premier 70%.",
              "Execution a la bougie suivante, frais+glissement inclus.",
              "ATTENTION: Kraken limite la recuperation a 720 bougies par intervalle;",
              "l'archive locale permet progressivement de tester plus longtemps.", ""]
    for name, p in pfs.items():
        output.append(f"{name:18} {p.valeur_liquidation(fin):8.2f} EUR, "
                      f"achats {p.achats}, ventes {p.ventes}, drawdown {p.perte_max:.2f}%")
    return "\n".join(output)


def tests_sans_reseau() -> None:
    # Marche synthetique, seulement pour tester la MECANIQUE, pas la rentabilite.
    ts = 1800000000
    bars = []
    for i in range(350):
        prix = 50000 + 4*i + 90*math.sin(i/9)
        bars.append(Bougie(ts+i*900, prix-3, prix+20, prix-20, prix))
    x = caracteristiques(bars, 200)
    copie = bars[:]
    copie[201] = Bougie(copie[201].ts, 1., 1., 1., 999999.)
    assert caracteristiques(copie, 200) == x, "Fuite d'information future"
    ia, _ = construire_ia(bars)
    assert 0 <= proba_hausse(ia, bars) <= 1
    sig = signaux(bars, ia)
    assert set(sig) == set(NOMS)
    p = Portefeuille()
    assert p.acheter(50000, 1)
    assert p.euros > 0 and p.bitcoin > 0 and p.achats == 1
    assert p.vendre(50200, 2) and p.ventes == 1 and p.bitcoin == 0
    assert p.frais_cumules > 0
    bt = backtest_recentre(bars)
    assert "conservation_30" in bt
    print("TESTS LOCAUX REUSSIS: pas de donnees futures, couts, transactions, IA, backtest.")
    print("Ce n'est PAS une preuve de rentabilite.")


def verrouiller():
    f = VERROU.open("a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Le robot tourne deja dans un autre terminal. Ne lance pas deux exemplaires.")
    f.seek(0)
    f.truncate(0)
    f.write(str(os.getpid()))
    f.flush()
    return f  # Conserver la reference jusqu'a la fin du programme.


def main():
    parser = argparse.ArgumentParser(description="Suivi Bitcoin/EUR 24h/24 en argent FICTIF")
    parser.add_argument("--test", action="store_true", help="Tests locaux, sans Internet")
    parser.add_argument("--backtest", action="store_true", help="Backtest de 7j environ, sans ordre reel")
    parser.add_argument("--once", action="store_true", help="Une iteration et sortie")
    args = parser.parse_args()
    if args.test:
        tests_sans_reseau()
        return
    if args.backtest:
        recentes = lire_bougies()
        conserver_bougies(recentes)
        archive = charger_bougies_archive()
        print(backtest_recentre(archive if len(archive) >= len(recentes) else recentes))
        return

    lock = verrouiller()
    _ = lock
    print("ROBOT PAPIER 24/7 - BTC/EUR - AUCUN ORDRE REEL POSSIBLE")
    print("Fichiers sauvegardes automatiquement dans:", BASE)
    print("Pour arreter : Ctrl+C. Pour relancer : meme commande, etat conserve.")
    etat = etat_charger()
    if etat is None and JOURNAL.exists():
        sys.exit("Journal existant sans etat sauvegarde : evite de melanger deux simulations. Sauvegarde les fichiers avant de repartir.")
    modele = None
    while True:
        try:
            etat, modele = observer_une_fois(etat, modele)
        except KeyboardInterrupt:
            print("\nArret demande. Etat sauvegarde apres chaque cycle.")
            break
        except Exception as exc:
            print(f"{horodatage()} | ERREUR RESEAU / DONNEES: {exc} | AUCUN ACHAT")
            if etat is not None:
                etat["erreurs"] = etat.get("erreurs", 0) + 1
                etat_sauver(etat)
        if args.once:
            break
        try:
            time.sleep(POLL_SECONDES)
        except KeyboardInterrupt:
            print("\nArret demande. Relance possible sans perdre les positions fictives.")
            break


if __name__ == "__main__":
    main()
