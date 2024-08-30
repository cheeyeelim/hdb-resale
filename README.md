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
2. All done!

# How to Develop

1. Setup `poetry` environment locally (i.e. not in Docker).
   1. `poetry lock`
   2. `poetry install -E dev -E test`
2. Update codes as needed.
   1. To update Python packages, modify `pyproject.toml`.
3. Continue to develop!

# Project Folder Structure

# Other Information
