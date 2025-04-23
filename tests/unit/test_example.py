from typing import Any, Dict


def test_example() -> None:
    """A dummy test example."""
    assert True


def test_example_with_data() -> None:
    """An example test with some data structure."""
    data: Dict[str, Any] = {"key": "value", "number": 42}
    assert "key" in data
    assert data["number"] == 42 