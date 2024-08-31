# hdb-resale

# Table of Contents

- [hdb-resale](#hdb-resale)
- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [How to Deploy](#how-to-deploy)
- [How to Develop](#how-to-develop)
- [Project Folder Structure](#project-folder-structure)
- [Other Information](#other-information)

# Overview

Code repository that handles data retrieval, processing and model training for HDB resale analytics.

It is intended to be installed as a package to power both the backend (i.e. Airflow pipeline) and the frontend (i.e. Dash) for HDB resale analytics tool.

# How to Deploy

1. Git push updated codes to repository.
   1. `git add .`
   2. `git commit -m "{commit message}"`
   3. `git push`
2. Go to other projects/repos that require this package and install as dependency.
   1. `poetry add git+https://github.com/cheeyeelim/hdb-resale.git`
3. All done!

# How to Develop

1. Setup `poetry` environment locally (i.e. not in Docker).
   1. `poetry lock`
   2. `poetry install -E dev -E test`
2. Update codes as needed.
   1. To update Python packages, modify `pyproject.toml`.
3. Continue to develop!

# Project Folder Structure

├── .env                            # environment variables
├── .gitignore
├── .pre-commit-config.yaml         # config for pre-commit
├── README.md
├── conf                            # hydra config for development purpose
│   └── hdb_resale_config.yaml
├── hdb_resale                      # hdb_resale module
│   ├── __init__.py
│   ├── api.py
│   ├── data.py
│   ├── model.py                    # ML models
│   ├── sql.py                      # data models
│   ├── task                        # task submodule containing all Airflow tasks
│   └── utils.py
├── notebooks                       # notebook for development purpose
│   └── hdb_resale
├── poetry.lock                     # poetry resolved Python dependencies
├── poetry.toml                     # config for poetry
├── pyproject.toml                  # config for poetry & Python dependencies
└── setup.cfg

# Other Information
