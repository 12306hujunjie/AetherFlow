# AetherFlow - 智能流式数据处理框架

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#测试)

一个现代化的 Python 数据流处理框架，专为构建可扩展、线程安全的数据处理管道而设计。通过流式接口（Fluent Interface）和智能依赖注入系统，让开发者能够以声明式的方式构建复杂的数据处理工作流。

## 🌟 核心特性

- 🔗 **声明式流程定义**: 通过链式 API 构建清晰的数据流
- 🧵 **线程安全**: 基于 ThreadLocalSingleton 的状态隔离机制
- 💉 **智能依赖注入**: 集成 dependency-injector 的 DI 系统
- ⚡ **并行处理**: 支持扇出/扇入模式的并行工作流
- 🔄 **自动重试**: 可配置的重试机制和异常处理
- 🛡️ **类型安全**: 完整的类型注解和 Pydantic 验证

## 🚀 快速开始

**开始构建您的高性能数据处理系统！** 🚀

```bash
# 安装依赖
pip install aetherflow
```
# 运行第一个示例
```python
from aetherflow import node
from pydantic import BaseModel

class SumResult(BaseModel):
    sum: int

class AverageResult(BaseModel):
    average: float

@node
def data_source(x: int, y: str):
    return {'numbers': list(range(x)), 'name': y}

@node
def calculate_sum(data: dict) -> SumResult:
    return SumResult(**{'sum': sum(data['numbers'])})

@node
def calculate_average(data: dict) -> AverageResult:
    numbers = data["numbers"]
    return AverageResult(**{'average': sum(numbers) / len(numbers)})

@node
def combine_results(parallel_results):
    """聚合并行处理结果"""

    sum_result = parallel_results['calculate_sum'].result
    avg_result = parallel_results['calculate_average'].result
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
flow = (data_source
    .fan_out_to([calculate_sum, calculate_average])
    .fan_in(combine_results))
then_flow = flow.branch_on({True: condition1, False: condition2}).then(then_node)

# average -> 5.0, sum -> 55, result -> False
result = flow(11, "2")
# condition2
then_result = then_flow(11, "2")

@node
def repeat_node(x: int) -> int:
    return x + 1


repeat_flow = repeat_node.repeat(3)
repeat_result = repeat_flow(1)
print(repeat_result)
# 4
```

## ⚡ 核心概念

### 节点 (Node)

节点是 AetherFlow 的基本执行单元，通过 [`@node`](src/aetherflow/__init__.py:699) 装饰器将普通函数转换为可链接的处理节点。

### 流式接口 (Fluent Interface)

通过方法链构建数据处理管道：

| 方法 | 功能 | 示例 |
|------|------|------|
| `.then()` | 顺序执行 | `node1.then(node2)` |
| `.fan_out_to()` | 并行扇出 | `source.fan_out_to([task1, task2])` |
| `.fan_in()` | 结果汇入 | `parallel_nodes.fan_in(aggregator)` |
| `.branch_on()` | 条件分支 | `condition.branch_on({True: path_a})` |
| `.repeat()` | 重复执行 | `processor.repeat(3)` |

### 依赖注入

AetherFlow 集成了线程安全的状态管理：

```python
from aetherflow import BaseFlowContext
from dependency_injector.wiring import Provide

@node
def stateful_processor(data, state: dict = Provide[BaseFlowContext.state]):
    """带状态的处理节点"""
    state['processed_count'] = state.get('processed_count', 0) + 1
    result = data['value'] * 2
    return {'result': result, 'count': state['processed_count']}

# 配置依赖注入
container = BaseFlowContext()
container.wire(modules=[__name__])
```

## 🏗️ 高级功能

### 并行处理

```python
from aetherflow import node
@node
def data_source(x: int):
    return {'numbers': list(range(x))}

@node
def calculate_sum(data):
    return {'sum': sum(data['numbers'])}

@node
def calculate_average(data):
    numbers = data['numbers']
    return {'average': sum(numbers) / len(numbers)}

@node
def combine_results(parallel_results):
    """聚合并行处理结果"""
    sum_result = parallel_results['calculate_sum']['sum']
    avg_result = parallel_results['calculate_average']['average']
    return {'sum': sum_result, 'average': avg_result}

# 构建并行管道
pipeline = (data_source
    .fan_out_to([calculate_sum, calculate_average])
    .fan_in(combine_results))

result = pipeline.run(1)
```

### 重试机制

```python
@node(
    retry_count=5,
    retry_delay=2.0,
    backoff_factor=2.0,       # 指数退避
    max_delay=30.0,
    exception_types=(ValueError, ConnectionError)
)
def network_request(data):
    """网络请求节点"""
    import requests
    response = requests.get(data['url'])
    return {'data': response.json()}
```

### 条件分支

```python
@node
def check_condition(data):
    return data['value'] > 10

@node
def high_value_processor(data):
    return {'result': 'high', 'value': data['value']}

@node
def low_value_processor(data):
    return {'result': 'low', 'value': data['value']}

# 条件分支管道
pipeline = check_condition.branch_on({
    True: high_value_processor,
    False: low_value_processor
})
```

## 🎯 使用场景

### ETL 数据处理
```python
etl_pipeline = (extract_from_database
    .then(transform_data)
    .then(validate_data)
    .then(load_to_warehouse))
```

### ai agent智能体工作流
```python
ml_pipeline = (preprocess_data
    .fan_out_to([model_a, model_b, model_c])
    .fan_in(ensemble_predictions)
    .then(postprocess_results))
```

## 📚 完整文档

详细的技术文档请参考：[AetherFlow技术文档.md](docs/AetherFlow技术文档.md)

文档包含：
- 📖 **完整的 API 参考**: 所有类和方法的详细说明
- 🛠️ **高级功能指南**: 状态管理、并发模式、错误处理
- ✨ **最佳实践**: 节点设计、性能优化、调试技巧
- 🎯 **实际应用案例**: ETL、ML、实时处理等场景


## 📊 性能特点

- 线程隔离模式在高并发下性能优势明显
- 智能重试机制减少临时故障影响
- 类型验证开销最小化
- 支持线程池和进程池两种并发模式

## 📈 技术栈

- **核心**: Python 3.10+
- **依赖注入**: dependency-injector
- **并发**: threading, concurrent.futures
- **类型支持**: typing, pydantic
- **验证**: Pydantic 2.11.7+

## 🤝 获取帮助

- **完整文档**: [AetherFlow技术文档.md](docs/AetherFlow技术文档.md)
- **代码示例**: 查看 `tests/` 目录
- **问题报告**: 提交 GitHub Issues

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---
