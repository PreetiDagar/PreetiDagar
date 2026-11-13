import pandas as pd

from rideguard import make_demo_data, normalize_status, prepare_dataset


def test_status_normalization():
    assert normalize_status("Completed") == "Completed"
    assert normalize_status("Cancelled by Driver") == "Cancelled"
    assert normalize_status("Incomplete") == "Incomplete"
    assert normalize_status("No Driver Found") == "No Driver Found"
    assert normalize_status("unknown status") is None


def test_demo_data_is_explicitly_structured():
    df = make_demo_data(120, seed=7)
    X, y, source = prepare_dataset(df)
    assert len(X) == len(y) == len(source)
    assert list(X.columns) == [
        "Hour", "Weekday", "Month", "Vehicle Type", "Pickup Location", "Drop Location"
    ]
    assert set(y.unique()).issubset(
        {"Completed", "Cancelled", "Incomplete", "No Driver Found"}
    )


def test_post_outcome_fields_are_not_predictors():
    df = make_demo_data(100, seed=9)
    df["Cancellation Reason"] = "test"
    df["Driver Rating"] = 4.8
    X, _, _ = prepare_dataset(df)
    assert "Cancellation Reason" not in X.columns
    assert "Driver Rating" not in X.columns
