"""
Computational Tools

Provides basic mathematical and computational tools for ReAct agents.
These tools demonstrate sync/async compatibility and parameter validation.
"""

import math
import secrets

from ..decorators import tool
from ..models import ToolCategory


@tool(
    name="calculator",
    description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
    category=ToolCategory.COMPUTE,
)
def calculator(operation: str, a: float, b: float) -> float:
    """
    Basic calculator for arithmetic operations.

    Args:
        operation: Operation type (add, subtract, multiply, divide)
        a: First number
        b: Second number

    Returns:
        Result of the arithmetic operation

    Raises:
        ValueError: If operation is not supported or division by zero
    """
    operation = operation.lower().strip()

    if operation in ["add", "+"]:
        return a + b
    elif operation in ["subtract", "-"]:
        return a - b
    elif operation in ["multiply", "*"]:
        return a * b
    elif operation in ["divide", "/"]:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(
            f"Unsupported operation: {operation}. Supported: add, subtract, multiply, divide"
        )


@tool(
    name="math_functions",
    description="Advanced mathematical functions (sin, cos, tan, log, exp, sqrt)",
    category=ToolCategory.COMPUTE,
)
def math_functions(function: str, value: float, base: float = math.e) -> float:
    """
    Advanced mathematical functions.

    Args:
        function: Math function name (sin, cos, tan, log, exp, sqrt, pow)
        value: Input value
        base: Base for logarithm or exponent for power function

    Returns:
        Result of the mathematical function
    """
    function = function.lower().strip()

    try:
        if function == "sin":
            return math.sin(value)
        elif function == "cos":
            return math.cos(value)
        elif function == "tan":
            return math.tan(value)
        elif function == "log":
            if base == math.e:
                return math.log(value)
            else:
                return math.log(value, base)
        elif function == "exp":
            return math.exp(value)
        elif function == "sqrt":
            return math.sqrt(value)
        elif function == "pow":
            return math.pow(value, base)
        else:
            raise ValueError(
                f"Unsupported function: {function}. "
                "Supported: sin, cos, tan, log, exp, sqrt, pow"
            )
    except (ValueError, OverflowError) as e:
        raise ValueError(f"Mathematical error in {function}({value}): {str(e)}") from e


@tool(
    name="statistics",
    description="Calculate basic statistics (mean, median, mode, std_dev) for a list of numbers",
    category=ToolCategory.COMPUTE,
)
def statistics(numbers: list[float], statistic: str) -> float | list[float]:
    """
    Calculate basic statistics for a list of numbers.

    Args:
        numbers: List of numbers to analyze
        statistic: Statistic to calculate (mean, median, mode, std_dev, all)

    Returns:
        Calculated statistic value or dictionary of all statistics
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics for empty list")

    statistic = statistic.lower().strip()

    # Calculate all statistics
    sorted_nums = sorted(numbers)
    n = len(numbers)

    # Mean
    mean = sum(numbers) / n

    # Median
    if n % 2 == 0:
        median = (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2
    else:
        median = sorted_nums[n // 2]

    # Mode (most frequent value)
    from collections import Counter

    counts = Counter(numbers)
    max_count = max(counts.values())
    mode_values = [num for num, count in counts.items() if count == max_count]

    # Standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = math.sqrt(variance)

    # Return requested statistic
    if statistic == "mean":
        return mean
    elif statistic == "median":
        return median
    elif statistic == "mode":
        return mode_values if len(mode_values) > 1 else mode_values[0]
    elif statistic in ["std_dev", "std", "standard_deviation"]:
        return std_dev
    elif statistic == "all":
        return {
            "mean": mean,
            "median": median,
            "mode": mode_values,
            "std_dev": std_dev,
            "count": n,
            "min": min(numbers),
            "max": max(numbers),
        }
    else:
        raise ValueError(
            f"Unsupported statistic: {statistic}. "
            "Supported: mean, median, mode, std_dev, all"
        )


@tool(
    name="random_number",
    description="Generate random numbers (integer or float) within specified range",
    category=ToolCategory.COMPUTE,
)
def random_number(
    min_value: float = 0.0,
    max_value: float = 1.0,
    count: int = 1,
    integer_only: bool = False,
    seed: int = None,
) -> float | int | list[float | int]:
    """
    Generate random numbers within a specified range.

    Args:
        min_value: Minimum value (inclusive)
        max_value: Maximum value (exclusive for integers, inclusive for floats)
        count: Number of random numbers to generate
        integer_only: Whether to generate integers only
        seed: Random seed for reproducible results

    Returns:
        Single number or list of numbers
    """
    if seed is not None:
        random.seed(seed)

    if count < 1:
        raise ValueError("Count must be at least 1")

    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    results = []
    for _ in range(count):
        if integer_only:
            # For integers, max_value is exclusive, use secrets for better security
            range_size = int(max_value) - int(min_value)
            value = int(min_value) + secrets.randbelow(range_size)
        else:
            # For floats, use SystemRandom which is cryptographically secure
            import random
            secure_random = random.SystemRandom()
            value = secure_random.uniform(min_value, max_value)
        results.append(value)

    return results[0] if count == 1 else results


@tool(
    name="number_converter",
    description="Convert numbers between different bases (binary, decimal, hexadecimal, octal)",
    category=ToolCategory.COMPUTE,
)
def number_converter(value: str, from_base: str, to_base: str) -> str:
    """
    Convert numbers between different number bases.

    Args:
        value: Number to convert as string
        from_base: Source base (binary, decimal, hex, octal)
        to_base: Target base (binary, decimal, hex, octal)

    Returns:
        Converted number as string
    """
    from_base = from_base.lower().strip()
    to_base = to_base.lower().strip()

    # Parse input value based on source base
    try:
        if from_base in ["binary", "bin", "2"]:
            decimal_value = int(value, 2)
        elif from_base in ["decimal", "dec", "10"]:
            decimal_value = int(value, 10)
        elif from_base in ["hexadecimal", "hex", "16"]:
            decimal_value = int(value, 16)
        elif from_base in ["octal", "oct", "8"]:
            decimal_value = int(value, 8)
        else:
            raise ValueError(f"Unsupported source base: {from_base}")

        # Convert to target base
        if to_base in ["binary", "bin", "2"]:
            return bin(decimal_value)
        elif to_base in ["decimal", "dec", "10"]:
            return str(decimal_value)
        elif to_base in ["hexadecimal", "hex", "16"]:
            return hex(decimal_value)
        elif to_base in ["octal", "oct", "8"]:
            return oct(decimal_value)
        else:
            raise ValueError(f"Unsupported target base: {to_base}")

    except ValueError as e:
        raise ValueError(f"Number conversion error: {str(e)}") from e
