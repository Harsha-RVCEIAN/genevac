# Installation and Execution Setup

## Prerequisites
- Python 3.10+
- `pip` or `conda`

## Step-by-Step Installation

```bash
# Clone the repository
git clone <repo-url>
cd genevac

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Running Tests
```bash
pytest tests/
```
