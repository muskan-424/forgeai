"""Builds demos/procurement/docs/spares_catalogue.csv deterministically (seed 2025).

The item templates (names, categories, price ranges, lead times) are the LLM-written part: see
demos/procurement/synthesis_prompts.md, prompt P3. This script only expands them into rows with a fixed
random seed so the catalogue is reproducible: `python demos/procurement/make_catalogue.py`."""
import csv
import random
from pathlib import Path

# supplier -> (part number prefix, category, [(description, min EUR, max EUR)])
TEMPLATES = {
    "Hartmann Marine GmbH": ("HM", "Engine and pumps", [
        ("Fuel injector, 6-cylinder main engine", 380, 520), ("Piston ring set, 400 mm bore", 640, 900),
        ("Cylinder liner, 400 mm bore", 2100, 2900), ("Turbocharger cartridge assembly", 5200, 7400),
        ("Lube oil pump, 45 m3/h", 1800, 2600), ("Fuel oil filter element", 35, 70),
        ("Cooling water pump seal kit", 120, 210), ("Camshaft bearing shell set", 310, 450),
        ("Exhaust valve spindle", 760, 1100), ("Starting air valve", 540, 780),
        ("Crankcase relief valve", 410, 590), ("Governor actuator", 3300, 4600),
        ("Fresh water generator ejector", 950, 1400), ("Oil mist detector sensor", 260, 380),
        ("Bilge pump impeller", 150, 260)]),
    "Pacific Valve Co.": ("PV", "Valves and fittings", [
        ("Gate valve DN50, bronze", 85, 140), ("Gate valve DN100, cast steel", 260, 410),
        ("Globe valve DN40, bronze", 95, 150), ("Butterfly valve DN150, wafer type", 190, 320),
        ("Check valve DN80, swing type", 120, 210), ("Ball valve DN25, stainless steel", 45, 90),
        ("Sea chest strainer DN200", 380, 560), ("Quick-closing valve DN50", 420, 640),
        ("Safety relief valve DN32", 310, 470), ("Flange gasket kit DN100", 18, 40),
        ("Pipe coupling DN80, flexible", 75, 130), ("Non-return valve DN65", 110, 190),
        ("Pressure reducing valve DN25", 340, 520), ("Expansion joint DN100", 230, 360),
        ("Hose coupling set, 50 mm", 28, 55)]),
    "Baltic Electric AB": ("BE", "Electrical and navigation", [
        ("Alternator AVR unit, 440 V", 880, 1300), ("Gyro compass sensor", 4800, 6900),
        ("VHF marine radio, DSC class D", 420, 640), ("GPS receiver, IEC 61108", 380, 580),
        ("Navigation light, LED masthead", 140, 230), ("Echo sounder transducer", 1200, 1800),
        ("Circuit breaker, 630 A", 560, 840), ("Emergency lighting battery pack", 95, 160),
        ("AIS transponder class A", 1900, 2800), ("Radar magnetron", 1500, 2300),
        ("Cable gland set, brass", 22, 48), ("Motor starter, 22 kW", 410, 620),
        ("Voyage data recorder capsule battery", 310, 470), ("Shaft generator controller", 3100, 4500),
        ("Anemometer sensor", 270, 410)]),
    "Orion Safety Ltd": ("OS", "Lifesaving and fire-fighting", [
        ("Lifejacket, SOLAS approved", 38, 65), ("Liferaft service kit, 25 person", 640, 920),
        ("EPIRB battery, 6 year", 210, 330), ("Fire extinguisher 9 kg, foam", 85, 140),
        ("SCBA cylinder, 6.8 L", 360, 520), ("Fireman outfit, complete", 780, 1100),
        ("Immersion suit, adult", 190, 290), ("Hydrostatic release unit", 55, 95),
        ("Lifebuoy with light and smoke signal", 70, 120), ("Fire hose, 20 m with couplings", 90, 150),
        ("CO2 cylinder, 45 kg refill", 280, 410), ("Rocket parachute flares, box of 4", 120, 190),
        ("Breathing apparatus compressor filter", 150, 240), ("Emergency escape breathing device", 160, 250),
        ("Fixed foam concentrate, 25 L", 110, 180)]),
}
WARRANTY = [12, 12, 12, 24, 24, 6]
STOCK = ["In stock", "In stock", "In stock", "Made to order", "Backordered"]


def main() -> None:
    rng = random.Random(2025)
    rows = []
    for supplier, (prefix, category, items) in TEMPLATES.items():
        used = set()
        for desc, lo, hi in items:
            while True:
                no = f"{prefix}-{rng.randint(1000, 9999)}"
                if no not in used:
                    used.add(no)
                    break
            price = round(rng.uniform(lo, hi), 0 if hi > 200 else 2)
            stock = rng.choice(STOCK)
            lead = {"In stock": rng.choice([3, 5, 7]), "Made to order": rng.choice([21, 28, 35, 45]),
                    "Backordered": rng.choice([42, 56, 70])}[stock]
            rows.append([no, desc, category, supplier, f"{price:.2f}", lead, stock, rng.choice(WARRANTY)])
    out = Path(__file__).parent / "docs" / "spares_catalogue.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["part_no", "description", "category", "supplier", "unit_price_eur", "lead_time_days",
                    "stock_status", "warranty_months"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
