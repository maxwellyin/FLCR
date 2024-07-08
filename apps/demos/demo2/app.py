from urllib.parse import urlencode
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

try:
    from . import network
except ImportError:
    import network


APP_DIR = Path(__file__).resolve().parent


app = FastAPI()
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def render_template(request: Request, template_name: str, **context):
    return templates.TemplateResponse(request, template_name, context)


@app.get("/", response_class=HTMLResponse, name="home")
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return render_template(request, "home.html")


@app.get("/demo", response_class=HTMLResponse, name="demo")
@app.get("/demo/input", response_class=HTMLResponse)
async def demo(request: Request):
    return render_template(request, "input.html")


@app.post("/demo")
@app.post("/demo/input")
async def demo_submit(request: Request, text: str = Form(...), submit_button: str = Form(...)):
    route_name = "outcome" if submit_button == "submit" else "outcome_cluster"
    query = urlencode({"text": text})
    redirect_url = f"{request.url_for(route_name)}?{query}"
    return RedirectResponse(url=redirect_url, status_code=303)


@app.get("/demo/outcome", response_class=HTMLResponse, name="outcome")
async def outcome(request: Request, text: str):
    candidates = network.recommend(text)
    return render_template(request, "outcome.j2", text=text, candidates=candidates)


@app.get("/demo/outcomeCluster", response_class=HTMLResponse, name="outcome_cluster")
async def outcome_cluster(request: Request, text: str):
    candidateClusters = network.recommendBatch(network.readRawString(text))
    return render_template(request, "outcomeCluster.j2", text=text, candidateClusters=candidateClusters)


@app.get("/author", response_class=HTMLResponse, name="author")
async def author(request: Request):
    return render_template(request, "author.html")
