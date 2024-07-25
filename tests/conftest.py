import pytest as pt
from pathlib import Path

TEST_DATA_DIR = Path(__file__).parent / "data"

TEST_HL_DATA_FILE_ID1 = 33659

@pt.fixture
def data_files_dir():
    return str(TEST_DATA_DIR)
