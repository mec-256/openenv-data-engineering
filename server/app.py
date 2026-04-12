"""
FastAPI server for the OpenEnv environment.
Exposes /health, /reset, /step endpoints and runs on port 7860.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="OpenEnv: Advanced Data Engineering Pipeline")

# Import environment components
from environment import DataEnv
from tasks import TASKS

# Store environment instance globally
env = DataEnv()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "service": "openenv-data-engineering-pipeline",
            "version": "2.0.0",
        },
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return JSONResponse(
        status_code=200,
        content={
            "message": "OpenEnv: Advanced Multi-Table Data Engineering Pipeline",
            "docs": "/docs",
            "health": "/health",
        },
    )


@app.post("/reset")
async def reset_environment(task_id: str = "easy_data_cleaning"):
    """
    Reset the environment to the initial state.
    Returns observation AND info (Gymnasium protocol).
    """
    try:
        obs = env.reset(task_id=task_id)
        # CRITICAL: Judge expects both observation and info
        return JSONResponse(
            status_code=200,
            content={
                "observation": {
                    "dataset_sample": obs.dataset_sample,
                    "columns": obs.columns,
                    "dtypes": obs.dtypes,
                    "total_rows": obs.total_rows,
                    "missing_values": obs.missing_values,
                    "task_description": obs.task_description,
                    "feedback": obs.feedback,
                    "debug_hints": obs.debug_hints,
                },
                "info": {},
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step_environment(action: dict):
    """
    Execute an action and return observation, reward, done, info.
    """
    try:

        class DummyAction:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
                if not hasattr(self, "action_type"):
                    self.action_type = "submit"

        action_obj = DummyAction(action)
        obs, reward, done, info = env.step(action_obj)

        return JSONResponse(
            status_code=200,
            content={
                "observation": {
                    "dataset_sample": obs.dataset_sample,
                    "columns": obs.columns,
                    "dtypes": obs.dtypes,
                    "total_rows": obs.total_rows,
                    "missing_values": obs.missing_values,
                    "task_description": obs.task_description,
                    "feedback": obs.feedback,
                    "debug_hints": obs.debug_hints,
                },
                "reward": {
                    "value": reward.value,
                    "message": reward.message,
                },
                "done": done,
                "info": info,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    """Return available tasks."""
    return JSONResponse(
        status_code=200,
        content={
            "tasks": [
                {
                    "id": task_id,
                    "description": TASKS[task_id].description,
                }
                for task_id in TASKS.keys()
            ]
        },
    )
