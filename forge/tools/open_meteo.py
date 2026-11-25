import requests

from .base import Manifest, Tool


class OpenMeteo(Tool):
    manifest = Manifest(
        name="open_meteo",
        description="Get current weather (temperature, wind) for a named place via the keyless Open-Meteo API.",
        args={"location": "place name, e.g. 'Rotterdam'"},
        when_needed="Only when the brief needs LIVE weather or sea/wind conditions. Never for policy or catalogue questions.",
        # unambiguous phrases only: bare 'forecast' (budget forecast), 'wind' (wind turbine) or 'temperature'
        # (storage temperature) would wrongly make an unrelated topic look like it needs a weather API
        strong_keywords=["weather", "weather forecast", "sea state", "wind speed"],
        keywords=["weather", "weather forecast", "wind speed", "sea state", "storm", "rainfall"],
    )

    def run(self, args: dict) -> str:
        loc = str(args.get("location", "")).strip()
        if not loc:
            return "error: location required"
        try:
            g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                             params={"name": loc, "count": 1}, timeout=15).json()
            if not g.get("results"):
                return f"error: place '{loc}' not found"
            r = g["results"][0]
            w = requests.get("https://api.open-meteo.com/v1/forecast",
                             params={"latitude": r["latitude"], "longitude": r["longitude"],
                                     "current_weather": "true"}, timeout=15).json()["current_weather"]
            return (f"{r['name']}, {r.get('country', '')}: {w['temperature']} C, "
                    f"wind {w['windspeed']} km/h (as of {w['time']})")
        except Exception as e:  # noqa: BLE001
            return f"error: weather service unavailable ({e})"
