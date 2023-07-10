from typing import Iterable
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

def regression_report(y_true:Iterable, y_pred:Iterable, digits:int = 2) -> str:
    """_summary_

    Args:
        y_true (Iterable): The true elements
        y_pred (Iterable): The predicted elements
        digits (int, optional): Number of digits to show in the report values. Defaults to 2.

    Returns:
        str: The regression report
    """
    assert len(y_true) == len(y_pred), print(f"Lenghts of arrays don't match, got {len(y_pred)} and {len(y_true)}")
    headers = ["mae", "mape", "mse", "r2", "support"]
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    lines = ("score", mae, mape, mse, r2, len(y_true))    
    width = len("score")
    head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
    report = head_fmt.format("", *headers, width=width)
    report += "\n\n"
    row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * len(lines[1:-1]) + " {:>9}\n"
    report += row_fmt.format(*lines, width=width, digits=digits)        
    report += "\n"
    return report