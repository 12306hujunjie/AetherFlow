from pydantic import BaseModel

from aetherflow import node


class SumResult(BaseModel):
    sum: int


class AverageResult(BaseModel):
    average: float


@node
def data_source(x: int, y: str):
    return {"numbers": list(range(x)), "name": y}


@node
def calculate_sum(data: dict) -> SumResult:
    return SumResult(**{"sum": sum(data["numbers"])})


@node
def calculate_average(data: dict) -> AverageResult:
    numbers = data["numbers"]
    return AverageResult(**{"average": sum(numbers) / len(numbers)})


@node
def combine_results(parallel_results):
    """聚合并行处理结果"""

    sum_result = parallel_results["calculate_sum"].result
    avg_result = parallel_results["calculate_average"].result
    return True if sum_result.sum == avg_result.average else False


@node
def condition1():
    return True


@node
def condition2():
    return False


@node
def then_node(condition: bool) -> str:
    return "condition1" if condition else "condition2"


# 构建flow
flow = data_source.fan_out_to([calculate_sum, calculate_average]).fan_in(
    combine_results
)
then_flow = flow.branch_on({True: condition1, False: condition2}).then(then_node)

# {'average': 0.0, 'sum': 0}
result = flow(11, "2")
then_result = then_flow(11, "2")


@node
def repeat_node(x: int) -> int:
    return x + 1


repeat_flow = repeat_node.repeat(3)
repeat_result = repeat_flow(1)
print(repeat_result)
# 4
pass
