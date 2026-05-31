"""test_loop.py — cold A/B; warm navigate→decode→read→rank with mocks. M5."""

import pytest

pytestmark = pytest.mark.skip(reason="M5: end-to-end loop with mocked codec+reader")
