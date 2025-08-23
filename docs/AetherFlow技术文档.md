# AetherFlow - 智能流式数据处理框架技术文档

## 概述

AetherFlow 是一个现代化的 Python 数据流处理框架，专为构建可扩展、线程安全的数据处理管道而设计。通过流式接口（Fluent Interface）和智能依赖注入系统，让开发者能够以声明式的方式构建复杂的数据处理工作流。

### 核心特性

- 🔗 **声明式流程定义**: 通过链式 API 构建清晰的数据流
- 🧵 **线程安全**: 基于 ThreadLocalSingleton 的状态隔离机制
- 💉 **智能依赖注入**: 集成 dependency-injector 的 DI 系统
- ⚡ **并行处理**: 支持扇出/扇入模式的并行工作流
- 🔄 **自动重试**: 可配置的重试机制和异常处理
- 🛡️ **类型安全**: 完整的类型注解和 Pydantic 验证

### 环境要求

- Python 3.10+
- dependency-injector >= 4.48.1
- pydantic >= 2.11.7

## 快速开始

### 安装

```bash
pip install dependency-injector pydantic
```

### 第一个示例

```python
from aetherflow import node

@node
def load_data(filename):
    """加载数据文件"""
    with open(filename, 'r') as f:
        data = f.read().strip().split('\n')
    return {'data': data, 'count': len(data)}

@node
def filter_data(load_data, min_length=3):
    """过滤数据"""
    data = load_data['data']
    filtered = [item for item in data if len(item) >= min_length]
    return {
        'filtered_data': filtered,
        'original_count': load_data['count'],
        'filtered_count': len(filtered)
    }

@node
def save_results(filter_data, output_file):
    """保存结果"""
    with open(output_file, 'w') as f:
        f.write('\n'.join(filter_data['filtered_data']))
    return {
        'saved_file': output_file,
        'processed_items': filter_data['filtered_count'],
        'success': True
    }

# 构建和执行管道
pipeline = load_data.then(filter_data).then(save_results)
result = pipeline.run({
    'filename': 'input.txt',
    'min_length': 3,
    'output_file': 'output.txt'
})
```

## 核心概念

### 1. 节点 (Node)

节点是 AetherFlow 的基本执行单元，通过 [`@node`](src/aetherflow/__init__.py:699) 装饰器将普通函数转换为可链接的处理节点。

```python
@node
def process_data(data: dict) -> dict:
    """处理数据的节点"""
    result = data['input'] * 2
    return {'output': result}
```

### 2. 流式接口 (Fluent Interface)

通过方法链构建数据处理管道：

| 方法 | 功能 | 示例 |
|------|------|------|
| [`.then()`](src/aetherflow/__init__.py:302) | 顺序执行 | `node1.then(node2)` |
| [`.fan_out_to()`](src/aetherflow/__init__.py:306) | 并行扇出 | `source.fan_out_to([task1, task2])` |
| [`.fan_in()`](src/aetherflow/__init__.py:315) | 结果汇入 | `parallel_nodes.fan_in(aggregator)` |
| [`.branch_on()`](src/aetherflow/__init__.py:329) | 条件分支 | `condition.branch_on({True: path_a})` |
| [`.repeat()`](src/aetherflow/__init__.py:333) | 重复执行 | `processor.repeat(3)` |

### 3. 依赖注入

AetherFlow 集成了 [`BaseFlowContext`](src/aetherflow/__init__.py:230) 提供线程安全的状态管理：

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

## 核心 API 参考

### [`@node` 装饰器](src/aetherflow/__init__.py:699)

将函数转换为 Node 实例，支持重试机制和依赖注入。

```python
@node(
    name=None,                    # 节点名称，用于调试
    retry_count=3,               # 最大重试次数
    retry_delay=1.0,             # 重试间隔（秒）
    exception_types=(Exception,), # 需要重试的异常类型
    backoff_factor=1.0,          # 退避因子
    max_delay=60.0,              # 最大重试延迟
    enable_retry=True            # 是否启用重试
)
def my_function(data):
    pass
```

### [`Node` 类](src/aetherflow/__init__.py:240)

节点的核心实现，支持各种组合模式。

**主要方法：**

- [`then(next_node)`](src/aetherflow/__init__.py:302): 顺序链接节点
- [`fan_out_to(nodes, executor="thread")`](src/aetherflow/__init__.py:306): 并行分发到多个节点
- [`fan_in(aggregator)`](src/aetherflow/__init__.py:315): 聚合并行结果
- [`branch_on(conditions)`](src/aetherflow/__init__.py:329): 条件分支
- [`repeat(times, stop_on_error=False)`](src/aetherflow/__init__.py:333): 重复执行

### [`BaseFlowContext` 类](src/aetherflow/__init__.py:230)

依赖注入容器，提供线程安全的状态管理。

```python
class BaseFlowContext(containers.DeclarativeContainer):
    state = providers.ThreadLocalSingleton(dict)        # 线程本地状态
    context = providers.ThreadLocalSingleton(dict)      # 线程本地上下文
    shared_data = providers.Singleton(dict)             # 全局共享数据
```

### 异常类型

AetherFlow 提供完整的异常体系：

- [`AetherFlowException`](src/aetherflow/__init__.py:31): 基础异常类
- [`NodeExecutionException`](src/aetherflow/__init__.py:40): 节点执行异常
- [`NodeRetryExhaustedException`](src/aetherflow/__init__.py:68): 重试耗尽异常
- [`NodeTimeoutException`](src/aetherflow/__init__.py:54): 超时异常

## 高级功能

### 1. 并行处理

#### 扇出/扇入模式

```python
@node
def data_source():
    return {'numbers': list(range(100))}

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

result = pipeline.run({})
```

#### 执行器配置

```python
# 线程池执行器（适合 I/O 密集型）
thread_pipeline = source.fan_out_to(
    [task1, task2, task3],
    executor="thread",
    max_workers=4
)

# 进程池执行器（适合 CPU 密集型）
process_pipeline = source.fan_out_to(
    [task1, task2, task3],
    executor="process",
    max_workers=2
)
```

### 2. 重试机制

#### 基本重试配置

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

#### [`RetryConfig` 类](src/aetherflow/__init__.py:105)

```python
from aetherflow import RetryConfig

config = RetryConfig(
    retry_count=3,
    retry_delay=1.0,
    exception_types=(ValueError,),
    backoff_factor=2.0,
    max_delay=60.0
)

# 检查是否应该重试特定异常
should_retry = config.should_retry(ValueError("test"))

# 计算重试延迟（支持指数退避）
delay = config.get_delay(attempt_number)
```

### 3. 状态管理

#### 线程隔离模式（推荐）

```python
from dependency_injector import providers

class IsolatedContext(BaseFlowContext):
    """每个线程独立状态"""
    state = providers.ThreadLocalSingleton(dict)
    service = providers.ThreadLocalSingleton(MyService)
```

#### 共享状态模式

```python
import threading

class SharedStateService:
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter

class SharedContext(BaseFlowContext):
    """线程间协调"""
    shared_service = providers.Singleton(SharedStateService)
```

### 4. 条件分支

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

### 5. 循环处理

```python
@node
def increment_processor(data):
    current = data.get('value', 0)
    return {'value': current + 1}

# 重复执行 5 次
repeated_pipeline = increment_processor.repeat(5)
result = repeated_pipeline.run({'value': 0})  # {'value': 5}

# 遇到错误停止
safe_pipeline = increment_processor.repeat(3, stop_on_error=True)
```

## 并行结果模型

[`ParallelResult`](src/aetherflow/__init__.py:16) 数据类用于封装并行执行的结果：

```python
@dataclass
class ParallelResult:
    node_name: str                # 节点名称
    success: bool                 # 执行是否成功
    result: Any = None           # 执行结果
    error: str | None = None     # 错误信息
    error_traceback: str | None = None  # 错误堆栈
    execution_time: float | None = None # 执行时间
```

并行执行的返回格式：

```python
{
    "node_name": {
        "node_name": "节点名称",
        "success": True/False,
        "result": "执行结果或None",
        "error": "错误信息或None",
        "error_traceback": "错误堆栈或None",
        "execution_time": "执行时间（秒）"
    }
}
```

## 最佳实践

### 1. 节点设计原则

- **单一职责**: 每个节点只负责一个特定的处理任务
- **纯函数**: 尽量避免副作用，便于测试和调试
- **明确接口**: 使用类型注解定义清晰的输入输出

```python
@node
def clean_text(data: dict) -> dict:
    """清理文本数据"""
    text = data['text'].strip().lower()
    words = text.split()
    cleaned_words = [word for word in words if word.isalpha()]
    return {
        'original_text': data['text'],
        'cleaned_text': ' '.join(cleaned_words),
        'word_count': len(cleaned_words)
    }
```

### 2. 错误处理策略

```python
@node(
    retry_count=3,
    retry_delay=1.0,
    exception_types=(requests.RequestException,),
    enable_retry=True
)
def robust_api_call(data):
    """健壮的 API 调用"""
    try:
        response = requests.get(data['url'], timeout=10)
        response.raise_for_status()
        return {'success': True, 'data': response.json()}
    except requests.Timeout:
        raise NodeTimeoutException("API调用超时", timeout_seconds=10)
    except requests.RequestException as e:
        raise NodeExecutionException("API调用失败", original_exception=e)
```

### 3. 状态使用指导

```python
@node
def process_with_state(data, state: dict = Provide[BaseFlowContext.state]):
    """正确使用状态的示例"""
    # 读取状态
    processed_count = state.get('processed_count', 0)

    # 处理数据
    result = process_data(data)

    # 更新状态
    state['processed_count'] = processed_count + 1
    state['last_result'] = result

    return result
```

### 4. 并发模式选择

**选择线程隔离模式的场景：**
- 独立任务处理
- 简单的并发需求
- 新手开发者
- 高并发场景

**选择共享状态模式的场景：**
- 需要线程间协调
- 内存使用敏感
- 复杂的状态共享需求

## 使用场景

### 1. ETL 数据处理

```python
etl_pipeline = (extract_from_database
    .then(transform_data)
    .then(validate_data)
    .then(load_to_warehouse))
```

### 2. 机器学习推理

```python
ml_pipeline = (preprocess_data
    .fan_out_to([model_a, model_b, model_c])
    .fan_in(ensemble_predictions)
    .then(postprocess_results))
```

### 3. 实时数据处理

```python
realtime_pipeline = (receive_events
    .fan_out_to([fraud_detection, sentiment_analysis])
    .fan_in(generate_alerts)
    .then(send_notifications))
```

### 4. 批量文件处理

```python
batch_pipeline = (scan_directory
    .fan_out_to([process_images, extract_metadata])
    .fan_in(combine_results)
    .then(save_manifest))
```

## 性能和监控

### 性能特点

- 线程隔离模式在高并发下性能优势明显（+38% @ 16线程）
- 智能重试机制减少临时故障影响
- 类型验证开销最小化

### 调试和日志

```python
import logging

# 启用 AetherFlow 日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('aetherflow')

@node(name="debug_processor")  # 指定节点名称便于调试
def debug_node(data):
    logger.info(f"处理数据: {data}")
    return process_data(data)
```

### 监控建议

- 使用 `ParallelResult.execution_time` 监控节点性能
- 通过状态记录关键指标
- 利用异常信息进行问题诊断

## 总结

AetherFlow 提供了一个强大而灵活的数据流处理框架，通过声明式的 API 和智能的状态管理，让开发者能够构建从简单到复杂的各种数据处理管道。关键优势包括：

- **易用性**: 流式接口让代码清晰易懂
- **可靠性**: 完善的重试和异常处理机制
- **性能**: 线程安全的并发处理能力
- **扩展性**: 灵活的依赖注入和状态管理
- **类型安全**: 完整的类型注解支持

无论是简单的数据转换还是复杂的并行处理工作流，AetherFlow 都能提供清晰、可维护的解决方案。
