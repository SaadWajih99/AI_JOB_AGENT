import uuid
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

app = FastAPI(title="AgenticApply - The Autonomous AI Job Search Engine")
templates = Jinja2Templates(directory="templates")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
