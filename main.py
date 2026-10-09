import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

import job_agent

app = FastAPI(title="AgenticApply - The Autonomous AI Job Search Engine")
templates = Jinja2Templates(directory="templates")
app.mount("/artifacts", StaticFiles(directory=str(job_agent.ARTIFACTS_DIR)), name="artifacts")


class JobSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    limit: int = Field(default=25, ge=1, le=60)


class AgentRunRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_n: int = Field(default=5, ge=1, le=20)
    auto_submit: bool = False


class ProfileUpdate(BaseModel):
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""


class ResumeUploadRequest(BaseModel):
    """Text prompt metadata accompanying an uploaded resume."""

    prompt: str = Field(..., min_length=1, description="Target-role search prompt")
    filename: str = Field(default="resume.pdf", min_length=1)
    resume_text: str = Field(default="", max_length=200_000)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/upload-resume")
async def upload_resume(body: ResumeUploadRequest) -> dict:
    """Mocked resume ingestion endpoint.

    Accepts the text prompt metadata (plus optional parsed resume text) and
    returns a canned success payload as if an agent fleet had been scheduled.
    """
    upload_id = f"upl_{uuid.uuid4().hex[:12]}"
    agent_session_id = f"agt_{uuid.uuid4().hex[:10]}"

    return {
        "success": True,
        "status": "queued",
        "upload_id": upload_id,
        "agent_session_id": agent_session_id,
        "applications_queued": 1000,
        "message": (
            "Resume staged successfully. The autonomous agent fleet is warming "
            "up - 1,000 tailored applications are queued."
        ),
        "received": {
            "filename": body.filename,
            "prompt": body.prompt,
            "resume_chars": len(body.resume_text),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/app")
async def agent_dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@app.post("/api/cv/upload")
async def cv_upload(file: UploadFile = File(...)) -> dict:
    """Parse an uploaded CV (PDF/DOCX/TXT) and store the extracted profile."""
    data = await file.read()
    filename = Path(file.filename or "resume.txt").name
    text = job_agent.extract_text(data, filename)
    (job_agent.UPLOADS_DIR / filename).write_bytes(data)
    profile = job_agent.parse_profile(text, filename)
    job_agent.save_profile(profile)
    return {
        "success": True,
        "profile": profile,
        "text_preview": text[:400].strip(),
        "engine": "llm" if job_agent._llm_config() else "rule-based",
    }


@app.get("/api/profile")
async def get_profile() -> dict:
    return {"profile": job_agent.load_profile()}


@app.put("/api/profile")
async def update_profile(body: ProfileUpdate) -> dict:
    profile = job_agent.load_profile() or {}
    profile.update(body.model_dump())
    job_agent.save_profile(profile)
    return {"success": True, "profile": profile}


@app.post("/api/jobs/search")
async def jobs_search(body: JobSearchRequest) -> dict:
    profile = job_agent.load_profile() or {}
    jobs = await job_agent.search_jobs(
        body.query, profile.get("skills", []), limit=body.limit
    )
    job_agent.save_jobs(jobs)
    return {"count": len(jobs), "jobs": jobs}


@app.get("/api/jobs")
async def jobs_list() -> dict:
    return {"jobs": job_agent.load_jobs()}


@app.post("/api/agent/run")
async def agent_run(body: AgentRunRequest):
    """SSE stream of live agent log lines while it tailors + applies."""
    queue: asyncio.Queue = asyncio.Queue()

    def emit(line: str, tone: str = "info") -> None:
        queue.put_nowait({"line": line, "tone": tone})

    async def runner() -> None:
        try:
            await job_agent.run_agent(body.query, body.top_n, body.auto_submit, emit)
        except Exception as exc:  # noqa: BLE001 - surface everything to the console
            emit(f"fatal error: {exc}", "error")
        finally:
            queue.put_nowait(None)

    task = asyncio.create_task(runner())

    async def stream():
        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"
        await task

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


@app.get("/api/health")
async def health() -> dict:
    profile = job_agent.load_profile()
    return {
        "steel_reachable": job_agent.steel_reachable(),
        "steel_url": job_agent.STEEL_BASE,
        "tailoring_engine": "llm" if job_agent._llm_config() else "rule-based",
        "profile_loaded": bool(profile),
        "profile_name": (
            f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
            if profile
            else ""
        ),
        "stored_jobs": len(job_agent.load_jobs()),
        "applications": len(job_agent.list_applications()),
    }


@app.get("/api/applications")
async def applications() -> dict:
    return {"applications": job_agent.list_applications()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
