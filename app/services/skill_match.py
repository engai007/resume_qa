"""Is a JD skill in a resume? The most important step of matching: 65% of the score. See docs/jd_matching_plan.md.

A skill is found when one of its names appears as a whole word in the resume's chunks (skills list OR any
project / experience bullet). Rules, each one there because the opposite gave a wrong match:
- whole word, case-insensitive: "Go" never matches inside "Google", "Java" never inside "JavaScript"
- aliases: "Postgres" = "PostgreSQL", "k8s" = "Kubernetes", ... (ALIASES)
- alternatives: "Kafka/RabbitMQ", "FastAPI or Django" -> found if any one is
- very short names ("Go", "R", "C", "JS"): only in the skills section, else "go live" in a bullet would be Go
- "C" never matches "C#" or "C++" (a # or + right after the name means another language)
- skills that are also English words ("REST", "React", "Spring", ...): only capitalised / upper case, so
  "the rest of the team" or "react to incidents" don't count (COMMON_WORDS)
"""
import re
from dataclasses import dataclass
from functools import cache

# each group = names of one skill. Keep it to real synonyms, not related tools ("AWS" is not "Lambda").
ALIASES = [
    ["postgresql", "postgres", "psql"],
    ["kubernetes", "k8s"],
    ["javascript", "js"],
    ["typescript", "ts"],
    ["golang", "go"],
    ["node.js", "nodejs", "node js"],
    ["react", "react.js", "reactjs"],
    ["vue", "vue.js", "vuejs"],
    ["c#", "csharp", "c sharp"],
    ["c++", "cpp"],
    ["mongodb", "mongo"],
    ["elasticsearch", "elastic search"],
    ["aws", "amazon web services"],
    ["gcp", "google cloud", "google cloud platform"],
    ["azure", "microsoft azure"],
    ["machine learning", "ml"],
    ["artificial intelligence", "ai"],
    ["natural language processing", "nlp"],
    ["large language models", "llm", "llms"],
    ["rest api", "rest apis", "restful", "restful apis"],  # not bare "rest": "the rest of the team"
    ["ci/cd", "cicd", "ci cd"],
    ["scikit-learn", "sklearn", "scikit learn"],
    ["rabbitmq", "rabbit mq"],
    ["github actions", "gh actions"],
    ["sql server", "mssql", "ms sql"],
]
SHORT = 2  # names this long or shorter: skills section only
# skills that are also everyday words: matched only as "Rest"/"REST" (title or upper case), never "rest"
COMMON_WORDS = {"rest", "react", "spring", "express", "swift", "rust", "go", "chef", "puppet", "ant", "make",
                "flask", "next", "nest", "ember", "meteor", "spark", "storm", "hive", "pig", "airflow", "kafka",
                "terraform", "ansible", "jenkins", "redis", "elastic", "node", "angular", "vue", "dart", "julia",
                "scala", "ruby", "pandas", "keras", "excel", "word", "office", "access", "tableau", "snowflake"}
SKILLS_FIRST = ("skills",)  # where to look first: evidence from the skills list is the clearest


@dataclass(frozen=True)
class Evidence:
    skill: str  # as written in the JD
    found_as: str  # the name that matched, e.g. "Postgres" for "PostgreSQL"
    section: str
    chunk_index: int


def alternatives(skill: str) -> list[str]:
    """"Kafka/RabbitMQ" -> ["Kafka", "RabbitMQ"]; "FastAPI or Django" -> ["FastAPI", "Django"].
    "CI/CD" and "TCP/IP" stay whole (both sides 2-3 letters: one term, not a choice)."""
    s = skill.strip()
    parts = re.split(r"\s+or\s+", s, flags=re.I)
    out = []
    for p in parts:
        bits = [b.strip() for b in p.split("/")]
        if len(bits) > 1 and not all(len(b) <= 3 for b in bits):
            out += bits
        else:
            out.append(p.strip())
    return [o for o in out if o]


def names(skill: str) -> list[str]:
    """Every name to look for: the skill itself + its aliases."""
    low = skill.lower()
    for group in ALIASES:
        if low in group:
            return [skill] + [g for g in group if g != low]
    return [skill]


@cache
def pattern(name: str) -> re.Pattern:
    """Whole word: no letter/digit before; no letter/digit/#/+ after ("C" not in "C#"; "c++", "node.js" work).
    Case-insensitive, except common words: "rest" -> only "Rest" or "REST"."""
    if name.lower() in COMMON_WORDS:
        spellings = "|".join(re.escape(s) for s in {name.title(), name.upper(), name[0].upper() + name[1:]})
        return re.compile(rf"(?<![A-Za-z0-9])(?:{spellings})(?![A-Za-z0-9#+])")
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9#+])", re.I)


def find_skill(skill: str, chunks: list[tuple[str, str]]) -> Evidence | None:
    """skill from the JD, chunks = make_chunks(resume) as (section, text). The first place it's found, skills
    section first. None if it isn't in the resume."""
    order = sorted(range(len(chunks)), key=lambda i: chunks[i][0] not in SKILLS_FIRST)  # stable: resume order
    for alt in alternatives(skill):
        for name in names(alt):
            short = len(name) <= SHORT
            pat = pattern(name)
            for i in order:
                section, text = chunks[i]
                if short and section not in SKILLS_FIRST:
                    continue
                if m := pat.search(text):
                    return Evidence(skill=skill, found_as=m.group(0), section=section, chunk_index=i)
    return None


def match_skills(skills: list[str], chunks: list[tuple[str, str]]) -> dict | None:
    """-> {"score": share found, "matched": [Evidence as dict], "missing": [skill]}, or None if the JD lists none
    (then this part is left out of the score instead of counting as 0)."""
    if not skills:
        return None
    matched, missing = [], []
    for skill in skills:
        ev = find_skill(skill, chunks)
        if ev:
            matched.append(ev.__dict__)
        else:
            missing.append(skill)
    return {"score": round(len(matched) / len(skills), 4), "matched": matched, "missing": missing}
