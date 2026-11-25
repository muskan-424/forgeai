from datetime import date, timedelta

from .base import Manifest, Tool


class CurrentDate(Tool):
    manifest = Manifest(
        name="current_date",
        description="Return today's date (YYYY-MM-DD, weekday), optionally shifted by a number of days.",
        args={"offset_days": "optional integer days to add (negative for past)"},
        when_needed="Questions that depend on today's date: deadlines, 'how many days until', notice periods, expiry.",
        keywords=["today", "deadline", "expire", "expiry", "days until", "due date", "current date", "notice period", "overdue"],
        strong_keywords=["today's date", "current date", "days until", "days left", "days remain"],
        provides_facts=False,
    )

    def run(self, args: dict) -> str:
        try:
            off = int(args.get("offset_days", 0) or 0)
        except (TypeError, ValueError):
            off = 0
        d = date.today() + timedelta(days=off)
        return f"{d.isoformat()} ({d.strftime('%A')})"
