Application entry points and interactive demos.

- `demos/demo1/`: earlier citation-recommendation demo
- `demos/demo2/`: grouped recommendation demo

Current demo apps run on FastAPI.

Typical startup commands from the repository root:

```bash
uvicorn apps.demos.demo1.app:app --reload
uvicorn apps.demos.demo2.app:app --reload
```

Notes:

- both apps render Jinja templates from their local `templates/` directories
- both apps load local checkpoints and FAISS index files from `resource/`
- both apps depend on sentence-transformer weights being available locally
