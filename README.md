# Model Pipeline

## Install

**Step One**: Create and activate the virtual environment for the current local development:
```bash
python -m venv venv
source venv/bin/activate
```

**Step Two**: Install the required packages:
```bash
pip install -r requirements.txt
```

**Step Three**: Create the notebook kernel for the currently activated virtual environment:
```bash
ipython kernel install --name "model-pipeline-env" --user
```

**Step Four**: Start notebook and choose the previously created kernel to run `model_pipeline.ipynb`:
```bash
jupyter notebook
```
