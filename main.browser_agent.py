import asyncio
import json
import re
from pathlib import Path
from typing import AsyncGenerator
from urllib.parse import quote_plus

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
STEEL_BASE = "http://localhost:3000"
STEEL_SESSIONS = f"{STEEL_BASE}/v1/sessions"

app = FastAPI(title="AI Browser Agent")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.cache = None


class RunRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def resolve_target_url(prompt: str) -> str:
    match = re.search(r"https?://[^\s<>\"']+", prompt, flags=re.IGNORECASE)
    if match:
        return match.group(0).rstrip(".,);]")
    stripped = prompt.strip()
    if re.match(r"^[\w.-]+\.[a-z]{2,}(/.*)?$", stripped, flags=re.IGNORECASE):
        return f"https://{stripped}"
    return f"https://www.google.com/search?q={quote_plus(stripped)}"


async def create_steel_session(http: aiohttp.ClientSession) -> str:
    async with http.post(STEEL_SESSIONS, json={}) as response:
        payload = await response.json(content_type=None)
        if response.status >= 400:
            raise RuntimeError(f"Session create failed ({response.status}): {payload}")
    session_id = payload.get("id") if isinstance(payload, dict) else None
    if not session_id:
        raise RuntimeError(f"Session response missing id: {payload}")
    return session_id


async def release_steel_session(http: aiohttp.ClientSession, session_id: str) -> None:
    url = f"{STEEL_SESSIONS}/{session_id}/release"
    try:
        async with http.post(url, json={}) as response:
            await response.read()
    except aiohttp.ClientError:
        pass


async def _extract_with_playwright(ws_url: str, target_url: str) -> dict:
    playwright = await async_playwright().start()
    browser = None
    try:
        browser = await playwright.chromium.connect_over_cdp(ws_url)
        context = browser.contexts[0]
        page = context.pages[0] if context.pages else await context.new_page()
        await page.goto(target_url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(1_200)
        title = await page.title()
        body_text = await page.inner_text("body")
        return {"title": title, "text": body_text, "url": page.url}
    finally:
        if browser is not None:
            await browser.close()
        await playwright.stop()


def extract_page_layout(ws_url: str, target_url: str) -> dict:
    # Playwright needs a Proactor loop on Windows; Uvicorn's loop cannot spawn it.
    return asyncio.run(_extract_with_playwright(ws_url, target_url))


async def run_browser_agent(prompt: str) -> AsyncGenerator[str, None]:
    target_url = resolve_target_url(prompt)
    yield sse({"type": "log", "message": f"> prompt received: {prompt}"})
    yield sse({"type": "log", "message": f"> resolved target: {target_url}"})
    yield sse({"type": "log", "message": "> creating Steel session at localhost:3000..."})

    session_id = None
    try:
        async with aiohttp.ClientSession() as http:
            session_id = await create_steel_session(http)
            yield sse({"type": "log", "message": f"> session id: {session_id}"})

            ws_url = f"ws://localhost:3000/v1/sessions/debug?sessionId={session_id}"
            yield sse({"type": "log", "message": f"> connecting Playwright over CDP: {ws_url}"})
            yield sse({"type": "log", "message": f"> navigating to {target_url}"})

            extracted = await asyncio.to_thread(extract_page_layout, ws_url, target_url)

            yield sse({"type": "log", "message": f"> page title: {extracted['title']}"})
            yield sse({"type": "log", "message": "> extracting body layout text..."})
            yield sse({"type": "log", "message": f"> extracted {len(extracted['text'])} characters"})
            yield sse(
                {
                    "type": "result",
                    "text": extracted["text"],
                    "title": extracted["title"],
                    "url": extracted["url"],
                }
            )
            yield sse({"type": "log", "message": "> run complete"})
    except Exception as exc:
        yield sse({"type": "error", "message": str(exc)})
        yield sse({"type": "log", "message": f"! error: {exc}"})
    finally:
        if session_id:
            async with aiohttp.ClientSession() as http:
                await release_steel_session(http, session_id)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/run")
async def run_agent(body: RunRequest):
    return StreamingResponse(
        run_browser_agent(body.prompt.strip()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
