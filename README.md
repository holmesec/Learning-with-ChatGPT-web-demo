# Learning with ChatGPT I - 02466 Project work

This repository contains the code for running and testing the pipeline produced for our work with the project "Learning with ChatGPT I". The repository also contains the code for a website developed as an interface for interacting with the piple and facilitating two A/B tests descripted in the report for the project.

### Files of interest

- `dev/pipeline.ipynb` Main file for running the pipeline developed for the project
- `dev/neural_search/result_generation.ipynb` Main file for generating NS results
- `dev/neural_search/q_w_chat.ipynb` Main file for interacting with ChatGPT for generating datasets

## Web

The code for a website that provides an interface for interacting with the pipeline can be found under the `web/` folder. To run the code you should follow the following steps:
**Installation:**

```bash
cd web
python -m venv .venv
source ./.venv/bin/activate #Windows: ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
yarn
```

**Running the flask app (development server):**

```bash
cd web
flask -A chatta run
```
