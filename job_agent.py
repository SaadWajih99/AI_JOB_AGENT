"""AgenticApply engine: CV parsing, job search, tailoring, and browser auto-apply.

Pipeline:  upload CV -> parse profile -> search real job boards -> score match
-> tailor resume/cover letter -> fill + submit applications via the local Steel
browser engine (Playwright over CDP) -> track everything in SQLite.

Tailoring uses an LLM when an API key is present (OPENAI_API_KEY, GROQ_API_KEY
or DEEPSEEK_API_KEY); otherwise a deterministic rule-based tailor is used.
"""

import asyncio
import html as html_lib
import json
import os
import re
import sqlite3
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path

import aiohttp
from playwright.async_api import async_playwright

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DB_PATH = BASE_DIR / "agent.db"
UPLOADS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

STEEL_BASE = "http://localhost:3000"
STEEL_SESSIONS = f"{STEEL_BASE}/v1/sessions"
STEEL_WS_BASE = STEEL_BASE.replace("http://", "ws://", 1)

# Greenhouse boards we pull live postings from (failures are skipped).
GH_SLUGS = (
    "figma", "databricks", "duolingo", "cloudflare", "mongodb", "datadog",
    "samsara", "robinhood", "affirm", "coursera", "instacart", "coinbase",
    "vercel", "airbnb", "pinterest", "scaleai", "andurilindustries", "gusto",
)


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                data_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                location TEXT,
                description TEXT,
                match_score REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS applications (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                title TEXT NOT NULL,
                company TEXT NOT NULL,
                url TEXT NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                match_score REAL,
                cover_letter TEXT,
                detail TEXT,
                screenshot TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def save_profile(data: dict) -> None:
    with _db() as conn:
        conn.execute(
            "INSERT INTO profile (id, data_json) VALUES (1, ?) "
            "ON CONFLICT(id) DO UPDATE SET data_json = excluded.data_json",
            (json.dumps(data),),
        )


def load_profile() -> dict | None:
    with _db() as conn:
        row = conn.execute("SELECT data_json FROM profile WHERE id = 1").fetchone()
    return json.loads(row["data_json"]) if row else None


def save_jobs(jobs: list[dict]) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM jobs")
        conn.executemany(
            "INSERT OR REPLACE INTO jobs "
            "(id, title, company, url, source, location, description, match_score) "
            "VALUES (:id, :title, :company, :url, :source, :location, :description, :match_score)",
            jobs,
        )


def load_jobs(job_ids: list[str] | None = None) -> list[dict]:
    with _db() as conn:
        if job_ids:
            marks = ",".join("?" for _ in job_ids)
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE id IN ({marks}) ORDER BY match_score DESC",
                job_ids,
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM jobs ORDER BY match_score DESC").fetchall()
    return [dict(r) for r in rows]


def record_application(**kw) -> None:
    kw.setdefault("id", f"app_{uuid.uuid4().hex[:10]}")
    kw.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    with _db() as conn:
        conn.execute(
            "INSERT INTO applications "
            "(id, job_id, title, company, url, source, status, match_score, cover_letter, detail, screenshot, created_at) "
            "VALUES (:id, :job_id, :title, :company, :url, :source, :status, :match_score, :cover_letter, :detail, :screenshot, :created_at)",
            kw,
        )


def list_applications() -> list[dict]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM applications ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


init_db()

# --------------------------------------------------------------------------- #
# CV parsing
# --------------------------------------------------------------------------- #
SKILL_LEXICON = {
    "python", "javascript", "typescript", "java", "go", "golang", "rust", "c++", "c#",
    "sql", "react", "node", "node.js", "next.js", "vue", "angular", "django", "flask",
    "fastapi", "spring", "graphql", "rest", "docker", "kubernetes", "aws", "gcp", "azure",
    "terraform", "ansible", "ci/cd", "git", "linux", "mongodb", "postgres", "postgresql",
    "mysql", "redis", "kafka", "rabbitmq", "elasticsearch", "snowflake", "dbt", "airflow",
    "spark", "hadoop", "pytorch", "tensorflow", "llm", "langchain", "rag", "mlops",
    "machine learning", "data analysis", "pandas", "numpy", "etl", "microservices",
    "system design", "devops", "sre", "security", "api design", "agile", "scrum",
    "product management", "figma", "javascript", "html", "css", "tailwind", "swift",
    "kotlin", "ruby", "rails", "php", "laravel", "salesforce", "excel", "tableau",
    "power bi", "looker", "customer success", "technical writing", "project management",
}

STOPWORDS = {
    "the", "and", "for", "with", "you", "our", "are", "will", "your", "this", "that",
    "have", "has", "from", "work", "who", "what", "when", "where", "about", "join", "can",
    "all", "into", "their", "they", "them", "team", "teams", "role", "job", "position",
    "company", "etc", "using", "use", "well", "per", "plus", "across", "within", "over",
    "under", "more", "other", "years", "year", "ability", "strong", "looking", "we", "a",
    "an", "of", "to", "in", "on", "at", "is", "be", "as", "or", "by", "it", "its", "youll",
}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")


def extract_text(data: bytes, filename: str) -> str:
    """Best-effort text extraction from PDF / DOCX / TXT resumes."""
    name = (filename or "").lower()
    if name.endswith(".pdf") or data[:5] == b"%PDF-":
        try:
            from io import BytesIO

            from pypdf import PdfReader

            reader = PdfReader(BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001 - degrade instead of failing the upload
            return f"[pdf parse error: {exc}]\n" + data.decode("utf-8", errors="ignore")
    if name.endswith(".docx") or data[:2] == b"PK":
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
                xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
            xml = re.sub(r"</w:p>", "\n", xml)
            return html_lib.unescape(re.sub(r"<[^>]+>", "", xml))
        except Exception:  # noqa: BLE001
            return data.decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#.]+", text.lower()) if len(t) > 1]


def parse_profile(text: str, filename: str = "") -> dict:
    """Extract name / contact / skills / bullet achievements from raw CV text."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    email = (_EMAIL_RE.search(text) or type("x", (), {"group": lambda *_: ""})()).group(0)
    phone_m = _PHONE_RE.search(text)
    phone = phone_m.group(0) if phone_m else ""

    first = last = ""
    for line in lines[:15]:
        if re.fullmatch(r"[A-Za-zÀ-ÿ'\- .]{2,40}", line) and "@" not in line and not any(
            w in line.lower() for w in ("resume", "curriculum", "engineer", "developer", "manager")
        ):
            parts = line.split()
            if len(parts) >= 2 and all(p[:1].isupper() for p in parts[:3]):
                first, last = parts[0], parts[1]
                break

    lowered = text.lower()
    skills = sorted(s for s in SKILL_LEXICON if s in lowered)
    bullets = [
        ln.lstrip("-*•● ")
        for ln in lines
        if (ln.startswith(("-", "*", "•", "●")) or (len(ln) > 40 and ". " not in ln[:5]))
        and any(ch.isdigit() for ch in ln)
    ][:25]
    if not bullets:
        bullets = [ln for ln in lines if len(ln) > 60][:15]

    return {
        "filename": filename,
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": phone,
        "skills": skills,
        "bullets": bullets,
        "raw_text": text[:20000],
        "text_chars": len(text),
    }


# --------------------------------------------------------------------------- #
# Job search (real boards)
# --------------------------------------------------------------------------- #
def _score_job(title: str, description: str, query_terms: set[str], skills: set[str]) -> float:
    title_toks = Counter(_tokens(title))
    desc_toks = set(_tokens(description))
    score = 0.0
    for term in query_terms:
        if term in title_toks:
            score += 2.0
        elif term in desc_toks:
            score += 0.8
    for skill in skills:
        if skill in title.lower():
            score += 1.2
        elif skill in desc_toks or skill in description.lower():
            score += 0.4
    norm = max(1.0, 2.0 * len(query_terms) + 0.4 * len(skills))
    return round(min(1.0, score / norm), 4)


def _strip_html(raw: str) -> str:
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw or "", flags=re.S | re.I)
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", text))


async def _gh_jobs(http: aiohttp.ClientSession, slug: str) -> list[dict]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true"
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return []
            payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return []
    out = []
    for job in payload.get("jobs", []):
        out.append(
            {
                "id": f"gh_{job['id']}",
                "title": (job.get("title") or "").strip(),
                "company": (job.get("company_name") or slug).strip().title(),
                "url": job.get("absolute_url", ""),
                "source": "greenhouse",
                "location": (job.get("location") or {}).get("name", ""),
                "description": _strip_html(job.get("content", ""))[:4000],
            }
        )
    return out


async def _remotive_jobs(http: aiohttp.ClientSession, query: str) -> list[dict]:
    url = f"https://remotive.com/api/remote-jobs?search={query[:60]}"
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return []
            payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return []
    out = []
    for job in payload.get("jobs", [])[:40]:
        out.append(
            {
                "id": f"rem_{job.get('id')}",
                "title": (job.get("title") or "").strip(),
                "company": (job.get("company_name") or "").strip(),
                "url": job.get("url", ""),
                "source": "remotive",
                "location": job.get("candidate_required_location", "remote"),
                "description": _strip_html(job.get("description", ""))[:4000],
            }
        )
    return out


async def _arbeitnow_jobs(http: aiohttp.ClientSession, page: int = 1) -> list[dict]:
    url = f"https://www.arbeitnow.com/api/job-board-api?page={page}"
    try:
        async with http.get(url, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                return []
            payload = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return []
    out = []
    for job in payload.get("data", [])[:40]:
        out.append(
            {
                "id": f"an_{job.get('slug', uuid.uuid4().hex[:8])}",
                "title": (job.get("title") or "").strip(),
                "company": (job.get("company_name") or "").strip(),
                "url": job.get("url", ""),
                "source": "arbeitnow",
                "location": job.get("location", ""),
                "description": _strip_html(job.get("description", ""))[:4000],
            }
        )
    return out


async def search_jobs(query: str, skills: list[str] | None = None, limit: int = 30) -> list[dict]:
    """Fan out across live boards, score against the query + known skills, dedupe."""
    query_terms = {t for t in _tokens(query) if t not in STOPWORDS}
    skill_set = set(skills or [])
    async with aiohttp.ClientSession() as http:
        results = await asyncio.gather(
            *[_gh_jobs(http, slug) for slug in GH_SLUGS],
            _remotive_jobs(http, query),
            _arbeitnow_jobs(http, 1),
            _arbeitnow_jobs(http, 2),
        )
    jobs: list[dict] = []
    seen_urls: set[str] = set()
    seen_roles: set[tuple[str, str]] = set()
    for batch in results:
        for job in batch:
            role_key = (job['title'].lower().strip(), job['company'].lower().strip())
            if not job['url'] or job['url'] in seen_urls or role_key in seen_roles:
                continue
            seen_urls.add(job['url'])
            seen_roles.add(role_key)
            job["match_score"] = _score_job(
                job["title"], job["description"], query_terms, skill_set
            )
            jobs.append(job)
    if query_terms:
        jobs = [j for j in jobs if j["match_score"] > 0.05]
    jobs.sort(key=lambda j: j["match_score"], reverse=True)
    return jobs[:limit]


# --------------------------------------------------------------------------- #
# Tailoring engine (LLM when a key exists, deterministic rules otherwise)
# --------------------------------------------------------------------------- #
def _jd_keywords(jd_text: str, top_n: int = 14) -> list[str]:
    toks = [t for t in _tokens(jd_text) if t not in STOPWORDS and len(t) > 2]
    freq = Counter(toks)
    skills_hit = [s for s in SKILL_LEXICON if s in jd_text.lower()]
    merged = list(dict.fromkeys(skills_hit + [w for w, _ in freq.most_common(top_n)]))
    return merged[:top_n]


def _pick_bullets(profile: dict, keywords: list[str], k: int = 6) -> list[str]:
    ranked = []
    for bullet in profile.get("bullets", []):
        low = bullet.lower()
        hits = sum(1 for kw in keywords if kw in low)
        digits = min(2, sum(ch.isdigit() for ch in bullet) // 3)
        ranked.append((hits * 2 + digits, bullet))
    ranked.sort(key=lambda x: x[0], reverse=True)
    picked = [b for _, b in ranked[:k]]
    return picked or profile.get("bullets", [])[:k]


def _llm_config() -> tuple[str, str, str] | None:
    """(base_url, api_key, model) for a known OpenAI-compatible provider, else None."""
    if os.environ.get("OPENAI_API_KEY"):
        return (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                os.environ["OPENAI_API_KEY"], "gpt-4o-mini")
    if os.environ.get("GROQ_API_KEY"):
        return ("https://api.groq.com/openai/v1", os.environ["GROQ_API_KEY"], "llama-3.3-70b-versatile")
    if os.environ.get("DEEPSEEK_API_KEY"):
        return ("https://api.deepseek.com/v1", os.environ["DEEPSEEK_API_KEY"], "deepseek-chat")
    return None


async def _llm_cover_letter(profile: dict, job: dict, keywords: list[str]) -> str | None:
    cfg = _llm_config()
    if not cfg:
        return None
    base_url, api_key, model = cfg
    prompt = (
        "Write a concise 3-paragraph cover letter (max 190 words) for the job below, "
        "grounded ONLY on the candidate's real experience. Weave in these keywords naturally: "
        f"{', '.join(keywords[:10])}.\n\n"
        f"CANDIDATE SKILLS: {', '.join(profile.get('skills', [])[:25])}\n"
        f"CANDIDATE HIGHLIGHTS:\n" + "\n".join("- " + b for b in profile.get("bullets", [])[:10]) +
        f"\n\nJOB: {job['title']} at {job['company']}\nDESCRIPTION: {job['description'][:1500]}"
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 450,
        "temperature": 0.7,
    }
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
        return data["choices"][0]["message"]["content"].strip()
    except Exception:  # noqa: BLE001 - always fall back to the rule-based letter
        return None


def _rule_cover_letter(profile: dict, job: dict, keywords: list[str]) -> str:
    name = " ".join(x for x in (profile.get("first_name"), profile.get("last_name")) if x) or "The candidate"
    matched = [s for s in profile.get("skills", []) if s in job["description"].lower()][:6]
    if not matched:
        matched = keywords[:5]
    highlights = _pick_bullets(profile, keywords, k=2)
    hi_sentence = " ".join(h.rstrip(".") for h in highlights)
    return (
        f"Dear {job['company']} Hiring Team,\n\n"
        f"I am excited to apply for the {job['title']} role. My background in "
        f"{', '.join(matched) if matched else 'software engineering'} maps directly to the "
        f"requirements in your posting, and I would bring hands-on delivery experience "
        f"from day one.\n\n"
        f"Relevant highlights: {hi_sentence}.\n\n"
        f"I would welcome the chance to discuss how my experience can help {job['company']} "
        f"ship faster. Thank you for your consideration.\n\n"
        f"Best regards,\n{name}"
    )


async def tailor(profile: dict, job: dict) -> dict:
    """Produce a tailored cover letter + ATS keyword report for one job."""
    keywords = _jd_keywords(job["title"] + " " + job["description"][:3000])
    cover = await _llm_cover_letter(profile, job, keywords) or _rule_cover_letter(profile, job, keywords)
    engine = "llm" if _llm_config() else "rule-based"
    try:
        jd = set(keywords)
        cv_tokens = set(_tokens(profile.get("raw_text", "")))
        ats = round(100 * len(jd & cv_tokens) / max(1, len(jd)), 1)
    except Exception:  # noqa: BLE001
        ats = 0.0
    return {
        "engine": engine,
        "keywords": keywords,
        "matched_skills": [s for s in profile.get("skills", []) if s in job["description"].lower()],
        "ats_score": ats,
        "cover_letter": cover,
    }


# --------------------------------------------------------------------------- #
# Browser auto-apply via Steel (Playwright over CDP)
# --------------------------------------------------------------------------- #
NAME_FIELDS = ["#first_name", "input[name='job_application[first_name]']", "input[name*='first' i]"]
LAST_FIELDS = ["#last_name", "input[name='job_application[last_name]']", "input[name*='last' i]"]
EMAIL_FIELDS = ["#email", "input[type='email']", "input[name*='email' i]"]
PHONE_FIELDS = ["#phone", "input[type='tel']", "input[name*='phone' i]"]
RESUME_FIELDS = ["#resume", "input[name='resume']", "input[type='file'][name*='resume' i]", "input[type='file']"]
COVER_FIELDS = ["#cover_letter", "textarea[name*='cover' i]", "#cover_letter_text"]
SUBMIT_FIELDS = ["#submit_app", "input[type='submit']", "button[type='submit']", "button:has-text('Submit application')"]


async def _fill_first(scopes, selectors: list[str], value: str) -> bool:
    for scope in scopes:
        for sel in selectors:
            try:
                loc = scope.locator(sel).first
                await loc.wait_for(state="visible", timeout=2500)
                await loc.fill(value, timeout=2500)
                return True
            except Exception:  # noqa: BLE001 - try next selector/scope
                continue
    return False


async def _upload_first(scopes, selectors: list[str], path: str) -> bool:
    for scope in scopes:
        for sel in selectors:
            try:
                loc = scope.locator(sel).first
                await loc.wait_for(state="attached", timeout=2500)
                await loc.set_input_files(path, timeout=2500)
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


async def _click_first(scopes, selectors: list[str]) -> bool:
    for scope in scopes:
        for sel in selectors:
            try:
                loc = scope.locator(sel).first
                await loc.wait_for(state="visible", timeout=2500)
                await loc.click(timeout=2500)
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


async def create_steel_session() -> tuple[str, str]:
    async with aiohttp.ClientSession() as http:
        async with http.post(
            STEEL_SESSIONS, json={}, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            payload = await resp.json(content_type=None)
            if resp.status >= 400 or not isinstance(payload, dict) or "id" not in payload:
                raise RuntimeError(f"Steel session failed ({resp.status}): {payload}")
    session_id = payload["id"]
    return session_id, f"{STEEL_WS_BASE}/v1/sessions/debug?sessionId={session_id}"


async def release_steel_session(session_id: str) -> None:
    try:
        async with aiohttp.ClientSession() as http:
            async with http.post(
                f"{STEEL_SESSIONS}/{session_id}/release", json={}, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                await resp.read()
    except aiohttp.ClientError:
        pass


async def apply_to_job(
    profile: dict,
    job: dict,
    tailored: dict,
    resume_path: str,
    auto_submit: bool,
    emit,
) -> dict:
    """Open the posting in a Steel browser, fill the form, optionally submit."""
    result = {"status": "failed", "detail": "", "screenshot": ""}
    session_id = None
    playwright = await async_playwright().start()
    browser = None
    try:
        session_id, ws_url = await create_steel_session()
        emit(f"steel session {session_id} established", "info")
        browser = await playwright.chromium.connect_over_cdp(ws_url)
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(job["url"], wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(1_500)
        scopes = [page] + [fr for fr in page.frames if fr is not page]

        body_text = (await page.inner_text("body")).lower()
        needs_login = any(k in job["url"].lower() for k in ("linkedin.com", "login"))
        captcha = any(k in body_text for k in ("recaptcha", "hcaptcha", "verify you are human"))
        if needs_login:
            result.update(status="blocked", detail="posting requires a logged-in account")
            return result

        full_name = " ".join(x for x in (profile.get("first_name"), profile.get("last_name")) if x).strip()
        if not await _fill_first(scopes, EMAIL_FIELDS, profile.get("email", "")):
            result.update(status="blocked", detail="no application form found on page")
            return result
        if full_name:
            await _fill_first(scopes, NAME_FIELDS, profile.get("first_name", full_name))
            await _fill_first(scopes, LAST_FIELDS, profile.get("last_name", ""))
        if profile.get("phone"):
            await _fill_first(scopes, PHONE_FIELDS, profile["phone"])
        uploaded = await _upload_first(scopes, RESUME_FIELDS, resume_path)
        emit(f"resume upload {'ok' if uploaded else 'not found (no file field)'}", "info")
        await _fill_first(scopes, COVER_FIELDS, tailored.get("cover_letter", "")[:4900])

        shot = ARTIFACTS_DIR / f"{job['id']}.png"
        await page.screenshot(path=str(shot), full_page=False)
        result["screenshot"] = str(shot.relative_to(BASE_DIR).as_posix())

        if captcha:
            result.update(status="blocked", detail="CAPTCHA detected - needs human verification")
            return result
        if not auto_submit:
            result.update(status="staged", detail="form filled (dry-run: submit disabled)")
            return result

        clicked = await _click_first(scopes, SUBMIT_FIELDS)
        if not clicked:
            result.update(status="blocked", detail="submit button not found")
            return result
        await page.wait_for_timeout(3_500)
        after = (await page.inner_text("body")).lower()
        shot2 = ARTIFACTS_DIR / f"{job['id']}_after.png"
        await page.screenshot(path=str(shot2), full_page=False)
        result["screenshot"] = str(shot2.relative_to(BASE_DIR).as_posix())
        if any(k in after for k in ("thanks for applying", "thank you for applying", "application received", "application submitted")):
            result.update(status="submitted", detail="application confirmed by employer portal")
        elif "captcha" in after or "hcaptcha" in after:
            result.update(status="blocked", detail="CAPTCHA appeared at submit time")
        else:
            result.update(status="submitted", detail="submit clicked - confirmation text not detected; verify screenshot")
        return result
    except Exception as exc:  # noqa: BLE001 - one failed job must not kill the run
        result.update(status="failed", detail=str(exc)[:300])
        return result
    finally:
        try:
            if browser is not None:
                await browser.close()
        finally:
            try:
                await playwright.stop()
            finally:
                if session_id:
                    await release_steel_session(session_id)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def steel_reachable() -> bool:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", 3000), timeout=3):
            return True
    except OSError:
        return False


def _tailored_resume_text(profile: dict, job: dict, tailored: dict) -> str:
    bullets = _pick_bullets(profile, tailored["keywords"], k=8)
    name = " ".join(x for x in (profile.get("first_name"), profile.get("last_name")) if x) or "Candidate"
    lines = [
        name.upper(),
        f"Target role: {job['title']} at {job['company']}",
        profile.get("email", ""),
        "",
        "SUMMARY",
        f"Tailored for this posting. ATS keyword coverage: {tailored['ats_score']}%.",
        f"Core keywords: {', '.join(tailored['keywords'][:10])}",
        "",
        "SELECTED ACHIEVEMENTS (tailored)",
    ]
    lines += [f"- {b}" for b in bullets]
    lines += ["", "SKILLS", ", ".join(profile.get("skills", [])[:30])]
    return "\n".join(lines)


async def run_agent(query: str, top_n: int, auto_submit: bool, emit) -> dict:
    """Search -> tailor -> apply. `emit(line, tone)` receives live log lines."""
    stats = {"found": 0, "processed": 0, "submitted": 0, "staged": 0, "blocked": 0, "failed": 0}
    profile = load_profile()
    if not profile:
        emit("no CV uploaded - upload a resume first", "error")
        return stats

    resume_path = UPLOADS_DIR / (profile.get("filename") or "resume.txt")
    if not resume_path.exists():
        files = sorted(UPLOADS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            emit("stored resume file missing - re-upload your CV", "error")
            return stats
        resume_path = files[0]

    emit(f"profile loaded: {profile.get('first_name') or '?'} {profile.get('last_name') or ''} ({len(profile.get('skills', []))} skills)", "info")
    if not profile.get("email"):
        emit("WARNING: no email found in CV - application forms may fail", "warn")

    emit(f"steel engine at {STEEL_BASE} - {'reachable' if steel_reachable() else 'NOT REACHABLE (docker start steel-browser)'}", "info" if steel_reachable() else "warn")

    emit(f"searching live boards for '{query}' ...", "info")
    jobs = await search_jobs(query, profile.get("skills", []), limit=max(top_n * 4, 20))
    stats["found"] = len(jobs)
    save_jobs(jobs)
    emit(f"found {len(jobs)} relevant postings (scored against your CV)", "ok")

    selected = jobs[:top_n]
    engine_used = "llm" if _llm_config() else "rule-based"
    emit(f"tailoring engine: {engine_used} - applying to top {len(selected)} (auto_submit={'ON' if auto_submit else 'OFF'})", "info")

    for index, job in enumerate(selected, start=1):
        emit(f"[{index}/{len(selected)}] {job['company']} - {job['title']} (match {round(job['match_score'] * 100)}%)", "info")
        tailored = await tailor(profile, job)
        artifact = ARTIFACTS_DIR / f"{job['id']}_resume.txt"
        artifact.write_text(_tailored_resume_text(profile, job, tailored), encoding="utf-8")
        emit(f"  tailored: ats {tailored['ats_score']}% | keywords: {', '.join(tailored['keywords'][:5])}", "ok")

        result = await apply_to_job(profile, job, tailored, str(resume_path), auto_submit, emit)
        record_application(
            job_id=job["id"], title=job["title"], company=job["company"], url=job["url"],
            source=job["source"], status=result["status"], match_score=job["match_score"],
            cover_letter=tailored["cover_letter"], detail=result["detail"], screenshot=result["screenshot"],
        )
        stats["processed"] += 1
        stats[result["status"]] = stats.get(result["status"], 0) + 1
        tone = {"submitted": "ok", "staged": "ok", "blocked": "warn", "failed": "error"}.get(result["status"], "info")
        emit(f"  result: {result['status'].upper()} - {result['detail']}", tone)
        await asyncio.sleep(2)

    emit(
        f"run complete: {stats['processed']} processed | {stats.get('submitted', 0)} submitted | "
        f"{stats.get('staged', 0)} staged | {stats.get('blocked', 0)} blocked | {stats.get('failed', 0)} failed",
        "done",
    )
    return stats
