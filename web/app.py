import asyncio
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import LivestreamAgent
from generator import continue_conversation, start_conversation, _build_image_part, _call_api, _parse_response, b64_to_bytes
from models import Requirements
from web import EventType, ProgressEvent

logger = logging.getLogger(__name__)

app = FastAPI(title="Livestream Image Generator")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: dict[str, asyncio.Queue] = {}


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.post("/api/generate")
async def generate(
    reference: UploadFile = File(...),
    products: list[UploadFile] = File(...),
    person: str = Form(...),
    background: str = Form(...),
    position: str = Form("桌前展示"),
    notes: str = Form(""),
    aspect_ratio: str = Form("9:16"),
    max_rounds: int = Form(5),
):
    job_id = str(uuid.uuid4())[:8]
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir = OUTPUT_DIR / job_id
    job_output_dir.mkdir(parents=True, exist_ok=True)

    ref_path = job_upload_dir / reference.filename
    ref_path.write_bytes(await reference.read())

    product_paths = []
    for p in products:
        p_path = job_upload_dir / p.filename
        p_path.write_bytes(await p.read())
        product_paths.append(str(p_path))

    description = f"- 人物：{person}\n- 背景：{background}\n- 商品位置：{position}"
    if notes.strip():
        description += f"\n- 补充要求：{notes}"

    queue: asyncio.Queue = asyncio.Queue()
    jobs[job_id] = queue

    loop = asyncio.get_event_loop()

    def run_agent():
        try:
            requirements = Requirements(description=description)
            agent = LivestreamAgent(output_dir=str(job_output_dir), max_rounds=max_rounds)
            agent.aspect_ratio = aspect_ratio

            def on_progress(event: ProgressEvent):
                loop.call_soon_threadsafe(queue.put_nowait, event)

            agent.run(str(ref_path), product_paths, requirements, on_progress=on_progress)
        except Exception as e:
            error_event = ProgressEvent(event=EventType.ERROR, round=0, data={"message": str(e)})
            loop.call_soon_threadsafe(queue.put_nowait, error_event)

    asyncio.get_event_loop().run_in_executor(None, run_agent)

    return {"job_id": job_id}


@app.post("/api/edit/{job_id}")
async def edit_image(job_id: str, instruction: str = Form(...)):
    """Edit the final image with a text instruction."""
    job_output_dir = OUTPUT_DIR / job_id
    final_path = job_output_dir / "final.png"
    if not final_path.exists():
        return JSONResponse({"error": "no final image found"}, status_code=404)

    import base64
    image_b64 = base64.b64encode(final_path.read_bytes()).decode()

    prompt = f"请根据以下指令修改这张图片：\n{instruction}\n\n其他部分保持不变，只做指定的修改。"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ],
            }
        ]
    }

    try:
        response = _call_api(payload, response_modalities=["image", "text"])
        image_bytes, text, _ = _parse_response(response)
        if image_bytes is None:
            return JSONResponse({"error": "no image in response"}, status_code=500)

        edit_count = len(list(job_output_dir.glob("edit_*.png")))
        edit_filename = f"edit_{edit_count}.png"
        edit_path = job_output_dir / edit_filename
        edit_path.write_bytes(image_bytes)

        return {"image": edit_filename, "text": text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    queue = jobs.get(job_id)
    if not queue:
        return JSONResponse({"error": "job not found"}, status_code=404)

    async def event_generator():
        while True:
            event = await queue.get()
            yield f"data: {event.to_json()}\n\n"
            if event.event in (EventType.COMPLETED, EventType.ERROR):
                del jobs[job_id]
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/image/{job_id}/{filename}")
async def get_image(job_id: str, filename: str):
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/png")


@app.get("/api/download/{job_id}/{filename}")
async def download_image(job_id: str, filename: str):
    path = OUTPUT_DIR / job_id / filename
    if not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/png", filename=filename)
