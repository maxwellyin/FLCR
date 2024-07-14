Demo 2 is the grouped recommendation demo.

Current status:

- web framework: FastAPI
- retrieval backend: FAISS
- model style: grouped or multi-category recommendation flow

Run from the repository root:

```bash
uvicorn apps.demos.demo2.app:app --reload
```

This demo expects local resources under `apps/demos/demo2/resource/` and local embedding/model assets referenced by `network.py`.
