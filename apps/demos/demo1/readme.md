Demo 1 is the earlier single-list citation recommendation demo.

Current status:

- web framework: FastAPI
- retrieval backend: FAISS
- model style: earlier non-grouped recommendation flow

Run from the repository root:

```bash
uvicorn apps.demos.demo1.app:app --reload
```

This demo expects local resources under `apps/demos/demo1/resource/` and local embedding/model assets referenced by `network.py`.
