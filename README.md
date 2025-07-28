# AetherFlow

一个基于流式接口的Python工作流编排框架，支持顺序执行、并行处理、条件分支和循环控制。

## 特性

- **流式接口 (Fluent Interface)**：使用方法链构建清晰、可读的工作流
- **依赖注入**：自动解析和注入函数参数
- **并行执行**：支持扇出/扇入模式的并行处理
- **条件分支**：基于运行时条件的动态分支选择
- **循环控制**：支持固定次数循环和条件性提前退出
- **状态管理**：不可变状态传递，避免并发冲突

## 快速开始

### 基本用法

```python
from aetherflow import node, AppContainer

# 定义节点函数
@node
def add_numbers(a: int, b: int) -> dict:
    result = a + b
    return {"sum": result}

@node
def multiply_by_two(sum: int) -> dict:
    result = sum * 2
    return {"doubled": result}

# 构建顺序工作流
flow = add_numbers.then(multiply_by_two)

# 执行工作流
result = flow.run({"a": 5, "b": 3})
print(result)  # {'a': 5, 'b': 3, 'sum': 8, 'doubled': 16}
```

### 并行执行

```python
@node
def task_a(sum: int) -> dict:
    return {"task_a_result": f"A processed {sum}"}

@node
def task_b(sum: int) -> dict:
    return {"task_b_result": f"B processed {sum}"}

@node
def combine_results(parallel_results: dict) -> dict:
    return {"combined": f"Combined: {parallel_results}"}

# 并行扇出/汇入
flow = (
    add_numbers
    .fan_out_to([task_a, task_b])
    .fan_in(combine_results)
)

result = flow.run({"a": 2, "b": 3})
```

### 条件分支

```python
@node
def check_even(sum: int) -> bool:
    return sum % 2 == 0

@node
def process_even() -> dict:
    return {"result": "Even number processed"}

@node
def process_odd() -> dict:
    return {"result": "Odd number processed"}

# 条件分支
flow = (
    add_numbers
    .then(check_even.branch_on({
        True: process_even,
        False: process_odd
    }))
)

result = flow.run({"a": 2, "b": 3})  # sum=5 (奇数) -> process_odd
```

### 循环控制

```python
from aetherflow import LoopControl

@node
def refine_content(content: str, iteration: int) -> dict:
    if iteration >= 3:  # 条件性退出
        return LoopControl.BREAK
    
    refined = content + " [refined]"
    return {"content": refined}

# 最多重复5次，但可以提前退出
flow = refine_content.repeat(5)

result = flow.run({"content": "Initial", "iteration": 0})
```

## API 参考

### 流式接口方法

| 方法 | 描述 | 示例 |
|------|------|------|
| `.then(node)` | 顺序执行 | `node_a.then(node_b)` |
| `.fan_out_to([nodes])` | 并行扇出 | `node.fan_out_to([a, b])` |
| `.fan_in(aggregator)` | 并行汇入 | `parallel_flow.fan_in(combine)` |
| `.branch_on({condition: node})` | 条件分支 | `check.branch_on({True: a, False: b})` |
| `.repeat(count)` | 固定次数循环 | `node.repeat(3)` |

### 循环控制

在循环中，节点函数可以返回 `LoopControl.BREAK` 来提前终止循环：

```python
from aetherflow import LoopControl

@node
def conditional_task(counter: int) -> dict:
    if counter <= 0:
        return LoopControl.BREAK  # 提前退出
    return {"counter": counter - 1}
```

## 依赖注入

AetherFlow 自动解析函数参数并从状态中注入相应的值：

```python
@node
def process_data(input_value: int, config: str = "default") -> dict:
    # input_value 将从状态中自动注入
    # config 有默认值，如果状态中没有则使用默认值
    return {"processed": f"{input_value} with {config}"}
```

## 许可证

MIT License