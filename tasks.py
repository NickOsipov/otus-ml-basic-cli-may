"""
Invoke tasks for the machine learning project.
"""

from invoke import task, Context

@task
def predict(
    ctx: Context, 
    input = "input_data.csv", 
    output = "predictions.csv", 
    weight = 1.0, 
    intercept = 0.0
):
    """
    Run the main script to make predictions using the machine learning model.
    """
    cmd = "python3 src/main.py"
    cmd += f" --input {input}"
    cmd += f" --output {output}"
    cmd += f" --weight {weight}"
    cmd += f" --intercept {intercept}"
    ctx.run(cmd)

@task
def sync(ctx: Context):
    """
    Sync venv and dependencies.
    """
    ctx.run("uv sync")
