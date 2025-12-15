# AI/ML Model Registry

A registry service for uploading, storing, downloading, querying, and rating AI/ML model
artifacts via a REST API and a web UI.

## Implementation Scope
The project implements all required baseline features and a limited subset of
extended (security track) requirements (per our approved renegotiation).

## Features

### Baseline functionality
Baseline features (compliant with the OpenAPI schema) include:
- `Upload`: upload registry artifacts
- `Download`: download artifacts
- `Rate`: rate models over various metrics
    (e.g., reproducibility, reviewedness, tree score, net score, etc.)
- `Model ingest`: request the ingestion of a public HuggingFace model
- `Enumerate`: fetch a directory of all models (regex capable)
- `Lineage graph`: report the lineage graph of a model
- `Size cost`: check the size of a model download
- `License check`: determine a GitHub project's license compatibility with a model's license
- `Observability`: report system health via both an API endpoint and web UI
- `Reset`: reset to empty registry state with default user

### Extended functionality - Security Track
Extended features (approved upon renegotiation; partially implemented) include:
- `User authentication`: Username/password authentication that generates access tokens.
    Token-based access control is partially implemented.

## Architecture
- **Frontend:** Web-based UI hosted on AWS S3
- **Backend:** Python (FastAPI) REST service
- **Storage:** Local disk storage with associated metadata
- **Deployment:** AWS EC2

## Configuration
- `LOG_LEVEL`: logging verbosity

## Team (Group 6)
- Asem Elenawy
- Jaeyun Kim
- Jun Lim
- Kendall Lanning
