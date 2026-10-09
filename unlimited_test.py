import asyncio

import aiohttp
from playwright.async_api import async_playwright


STEEL_BASE_URL = "http://localhost:3000"
STEEL_WS_BASE_URL = STEEL_BASE_URL.replace("http://", "ws://", 1)
SESSIONS_URL = f"{STEEL_BASE_URL}/v1/sessions"
TARGET_URL = "https://ycombinator.com"


async def create_session(http: aiohttp.ClientSession) -> str:
    try:
        async with http.post(SESSIONS_URL, json={}) as response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(
                    f"Steel session creation failed ({response.status}): {payload}"
                )
    except aiohttp.ClientError as exc:
        raise RuntimeError(
            f"Could not reach Steel at {STEEL_BASE_URL}: {exc}. "
            "Is the local Docker container running?"
        ) from exc

    session_id = payload.get("id") if isinstance(payload, dict) else None
    if not session_id:
        raise RuntimeError(f"Steel session response did not include an id: {payload}")
    return session_id


async def release_session(http: aiohttp.ClientSession, session_id: str) -> None:
    async with http.post(f"{SESSIONS_URL}/{session_id}/release", json={}) as response:
        if response.status >= 400:
            detail = await response.text()
            raise RuntimeError(
                f"Steel session release failed ({response.status}): {detail}"
            )
        await response.read()


HEADLINE_SELECTORS = (
    "article h1, article h2, article h3, article h4, article a",
    "main h2, main h3, main h4",
    "main a",
)

# Keep only link-backed text of a reasonable length: article headlines are
# clickable, while plain section labels ("Knowledge & News", "Video", ...) are not.
_EXTRACT_JS = r"""
els => {
    const out = [];
    const seen = new Set();
    for (const el of els) {
        const text = (el.innerText || "").replace(/\s+/g, " ").trim();
        if (text.length < 10) continue;
        if (!el.closest("a") && !el.querySelector("a")) continue;
        if (seen.has(text)) continue;
        seen.add(text);
        out.push(text);
    }
    return out;
}
"""


async def _headline_texts(page, selector: str) -> list[str]:
    return await page.eval_on_selector_all(selector, _EXTRACT_JS)


async def extract_headlines(page) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for selector in HEADLINE_SELECTORS:
        for text in await _headline_texts(page, selector):
            if text not in seen:
                seen.add(text)
                collected.append(text)
        if len(collected) >= 5:
            break

    if len(collected) < 5:
        sample = await page.eval_on_selector_all(
            "h1, h2, h3, h4, a",
            r"""
            els => els
                .map(e => (e.innerText || "").replace(/\s+/g, " ").trim())
                .filter(t => t)
                .slice(0, 25)
            """,
        )
        raise RuntimeError(
            "Could not find five article headlines on the page. "
            f"Visible headings/links sample: {sample}"
        )
    return collected[:5]


async def main() -> None:
    async with aiohttp.ClientSession() as http:
        session_id = await create_session(http)
        playwright = await async_playwright().start()
        browser = None
        try:
            debug_ws_url = (
                f"{STEEL_WS_BASE_URL}/v1/sessions/debug?sessionId={session_id}"
            )
            browser = await playwright.chromium.connect_over_cdp(debug_ws_url)
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(1_500)

            headlines = await extract_headlines(page)
            print("Top 5 Y Combinator article headlines:")
            for index, headline in enumerate(headlines, start=1):
                print(f"{index}. {headline}")
        finally:
            try:
                if browser is not None:
                    await browser.close()
            finally:
                try:
                    await playwright.stop()
                finally:
                    await release_session(http, session_id)


if __name__ == "__main__":
    asyncio.run(main())
