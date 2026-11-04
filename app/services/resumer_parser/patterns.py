"""Regexes and symbol sets shared by the parsing steps."""
import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d\s-]{8,14}\d")
LINKEDIN = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?", re.I)
GITHUB = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[\w-]+/?", re.I)

BULLETS = "•◦∗▪●○■–*·- "
BULLET_RE = re.compile(r"^([•◦∗▪●○■–*·-]+)\s*")

MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = rf"(?:{MONTH}\s*['’]?\s*\d{{2,4}}|\d{{1,2}}/\d{{4}}|\d{{4}})"
DATE_RANGE = re.compile(
    rf"{DATE}\s*(?:-|–|—|to)\s*(?:{DATE}|present|current|now|till date)", re.I
)

SMALL_WORDS = {"and", "of", "&", "the", "in", "for"}
