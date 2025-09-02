# AetherFlow 技术文档

现代化Python数据流处理框架，通过智能异步适配和声明式API，让复杂工作流构建变得简单优雅。

## 目录

1. [快速开始](#快速开始)
2. [核心功能](#核心功能)
3. [完整示例](#完整示例)
4. [最佳实践](#最佳实践)
5. [API速查](#api速查)

---

## 快速开始

### 安装

```bash
# 使用PDM（推荐）
pdm add aetherflow

# 或使用pip
pip install aetherflow
```

**环境要求**: Python 3.10+ | 核心依赖: `dependency-injector`, `pydantic`

### 第一个工作流

```python
from aetherflow import node
import asyncio

@node
def extract_data(source: str) -> dict:
    return {"raw": f"data from {source}", "count": 100}

@node
async def transform_data(data: dict) -> dict:
    await asyncio.sleep(0.1)  # 异步操作
    return {"processed": data["raw"].upper(), "total": data["count"] * 2}

@node
def load_data(data: dict) -> str:
    return f"✅ 处理完成: {data['processed']} (总计: {data['total']})"

# 构建工作流 - 自动处理sync/async混合
pipeline = extract_data.then(transform_data).then(load_data)

# 执行
async def main():
    result = await pipeline("database")
    print(result)  # ✅ 处理完成: DATA FROM DATABASE (总计: 200)

asyncio.run(main())
```

### 核心概念

- **@node装饰器**: 将函数转为可组合的处理单元
- **智能异步**: 自动处理同步/异步函数混合调用
- **fluent接口**: `.then()`, `.fan_out_to()`, `.fan_in()` 等链式调用
- **并行处理**: 支持多种执行器的扇出/扇入模式
- **重试机制**: 基于异常分类的智能重试
- **状态管理**: 线程安全的依赖注入容器

---

## 核心功能

### 1. 顺序连接 (`.then()`)

**用途**: 将节点按顺序连接，前一个节点输出作为后一个节点输入。

```python
from aetherflow import node

@node
def extract_data(source: str) -> dict:
    return {"raw_data": f"data from {source}"}

@node
def transform_data(data: dict) -> dict:
    return {"transformed": data["raw_data"].upper()}

# 构建ETL管道
etl_pipeline = extract_data.then(transform_data)
result = etl_pipeline("database")  # {"transformed": "DATA FROM DATABASE"}
```

### 2. 并行扇出 (`.fan_out_to()`)

**用途**: 将输出广播给多个并行节点执行。

```python
@node
def source_data() -> dict:
    return {"value": 10}

@node
def task_multiply(data: dict) -> int:
    return data["value"] * 2

@node
def task_add(data: dict) -> int:
    return data["value"] + 5

# 并行执行
parallel_flow = source_data.fan_out_to([task_multiply, task_add])
results = parallel_flow()  # 返回ParallelResult字典
```

**执行器选择**:
- `"auto"`: 智能选择（推荐）
- `"async"`: 协程池，适合I/O密集任务
- `"thread"`: 线程池，适合CPU密集任务

### 3. 结果聚合 (`.fan_in()`)

```python
@node
def aggregate_results(parallel_results: dict) -> dict:
    successful = [r.result for r in parallel_results.values() if r.success]
    return {"total": sum(successful), "count": len(successful)}

# 扇出后聚合
flow = source_data.fan_out_to([task_multiply, task_add]).fan_in(aggregate_results)
result = flow()  # {"total": 35, "count": 2}
```

### 4. 条件分支 (`.branch_on()`)

```python
@node
def evaluate_score(data: dict) -> str:
    """评估分数，返回pass或fail"""
    score = data["score"]
    return "pass" if score >= 60 else "fail"

@node
def handle_pass(data: dict) -> dict:
    """处理通过情况"""
    return {"status": "通过", "score": data["score"]}

@node
def handle_fail(data: dict) -> dict:
    """处理未通过情况"""
    return {"status": "不通过", "action": "重考"}

# 条件分支：基于evaluate_score的返回值选择分支
grading_flow = evaluate_score.branch_on({
    "pass": handle_pass,    # 当evaluate_score返回"pass"时执行
    "fail": handle_fail     # 当evaluate_score返回"fail"时执行
})

# 使用示例
result = grading_flow({"score": 75})  # {"status": "通过", "score": 75}
```

### 5. 重复执行 (`.repeat()`)

```python
@node
def iterative_improve(data: dict) -> dict:
    value = data.get("value", 0)
    return {"value": value * 1.1 + 1}  # 每次增长10%+1

# 重复5次
iterative_flow = iterative_improve.repeat(5)
result = iterative_flow({"value": 10})  # 经过5次迭代后的结果
```

### 6. 智能异步适配

**核心价值**: 自动处理同步/异步函数混合，零配置。

```python
@node
def sync_process(data: str) -> dict:
    return {"result": data.upper()}

@node
async def async_fetch(data: dict) -> dict:
    await asyncio.sleep(0.1)
    return {"fetched": f"async_{data['result']}"}

# 自动处理混合执行
mixed_flow = sync_process.then(async_fetch)
result = await mixed_flow("hello")  # 框架自动协调sync/async
```

### 7. 重试机制

**智能重试**: 基于异常分类的差异化策略。

```python
from aetherflow import RetryConfig

class TemporaryError(Exception):
    def __init__(self, msg):
        super().__init__(msg)
        self.retryable = True  # 可重试

@node(retry_config=RetryConfig(
    retry_count=3,
    retry_delay=1.0,
    backoff_factor=2.0,  # 指数退避
    exception_types=(ConnectionError, TemporaryError)
))
async def resilient_call(url: str) -> dict:
    # 框架自动重试
    return await fetch_data(url)
```

### 8. 状态管理

**线程安全**: 支持依赖注入的状态管理。

```python
from aetherflow import BaseFlowContext
from dependency_injector.wiring import Provide

container = BaseFlowContext()
container.wire(modules=[__name__])

@node
def stateful_processor(
    data: dict,
    state: dict = Provide[container.state]  # 线程本地状态
) -> dict:
    state['count'] = state.get('count', 0) + 1
    return {"processed": data, "count": state['count']}
```

### 9. 组合使用

所有核心功能可自由组合构建复杂流程：

```python
# 完整的数据处理流水线
complex_pipeline = (
    extract_data                     # 提取数据
    .then(evaluate_score)           # 评估条件
    .branch_on({                    # 条件分支
        "pass": (
            handle_pass             # 处理通过
            .fan_out_to([           # 并行操作
                save_record,
                send_notification
            ])
        ),
        "fail": handle_fail         # 处理失败
    })
)
```

---

## 完整示例

### ETL数据管道

演示异步提取、同步转换、条件分支的混合流程：

```python
from aetherflow import node
import asyncio

@node
async def extract_data(source: str) -> list:
    """异步数据提取"""
    await asyncio.sleep(0.1)
    return [{"id": 1, "value": "100"}, {"id": 2, "value": "invalid"}]

@node
def transform_data(data: list) -> list:
    """同步数据转换"""
    return [{"id": item["id"], "processed": float(item["value"])}
            for item in data if item["value"].isdigit()]

@node
def check_quality(data: list) -> str:
    """质量检查，返回质量等级"""
    return "high" if len(data) > 0 else "low"

@node
async def load_high_quality(data: list) -> dict:
    """处理高质量数据"""
    await asyncio.sleep(0.1)
    return {"loaded": len(data), "status": "success"}

@node
def handle_low_quality(data: list) -> dict:
    """处理低质量数据"""
    return {"loaded": 0, "status": "rejected", "reason": "low quality"}

# 使用状态管理保持数据流
from aetherflow import BaseFlowContext
from dependency_injector.wiring import Provide

@node
def store_data(data: list, state: dict = Provide[BaseFlowContext.state]) -> str:
    """存储数据并返回质量评估"""
    state["processed_data"] = data
    return "high" if len(data) > 0 else "low"

@node
async def load_stored_data(quality: str, state: dict = Provide[BaseFlowContext.state]) -> dict:
    """从状态加载数据进行处理"""
    data = state["processed_data"]
    await asyncio.sleep(0.1)
    return {"loaded": len(data), "status": "success"}

@node
def reject_stored_data(quality: str, state: dict = Provide[BaseFlowContext.state]) -> dict:
    """拒绝低质量数据"""
    return {"loaded": 0, "status": "rejected", "reason": "low quality"}

# ETL管道：异步→同步→条件分支→异步
etl_flow = (
    extract_data
    .then(transform_data)
    .then(store_data)
    .branch_on({
        "high": load_stored_data,
        "low": reject_stored_data
    })
)

# 执行示例
async def run_etl():
    # 初始化依赖注入容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])

    result = await etl_flow("database")
    print(result)  # {"loaded": 1, "status": "success"}

asyncio.run(run_etl())
```

### 并发处理示例

```python
@node
def generate_tasks() -> list:
    """生成任务列表"""
    return [{"task_id": i, "data": f"item_{i}"} for i in range(5)]

@node
def process_task(task: dict) -> dict:
    """处理单个任务"""
    import time
    time.sleep(0.1)  # 模拟处理时间
    return {"task_id": task["task_id"], "result": task["data"].upper()}

@node
def collect_results(results: dict) -> dict:
    """收集处理结果"""
    successful = [r.result for r in results.values() if r.success]
    return {
        "total_tasks": len(results),
        "successful": len(successful),
        "results": successful
    }

# 并发处理流程
concurrent_flow = (
    generate_tasks
    .fan_out_to([process_task] * 3, executor="thread")  # 并行处理
    .fan_in(collect_results)
)

result = concurrent_flow()
print(f"处理了 {result['total_tasks']} 个任务，成功 {result['successful']} 个")
```

---

## 最佳实践

### 节点设计原则

#### 1. 单一职责
每个节点只负责一个明确的处理任务。

```python
# ✅ 好的设计
@node
def extract_user_data(user_id: int) -> dict:
    """只负责提取用户数据"""
    return get_user_from_db(user_id)

@node
def validate_user_data(user_data: dict) -> dict:
    """只负责验证用户数据"""
    if not user_data.get("email"):
        raise ValueError("Email is required")
    return user_data

# ❌ 避免的设计
@node
def extract_and_validate_user(user_id: int) -> dict:
    """职责过多，难以测试和重用"""
    user_data = get_user_from_db(user_id)
    if not user_data.get("email"):
        raise ValueError("Email is required")
    return user_data
```

#### 2. 类型注解
始终为节点函数提供完整的类型注解。

```python
from typing import List, Dict
from pydantic import BaseModel

class UserData(BaseModel):
    id: int
    name: str
    email: str

@node
def process_users(user_ids: List[int]) -> List[UserData]:
    """处理用户列表，返回用户数据"""
    return [get_user_data(uid) for uid in user_ids]
```

#### 3. 错误处理
合理设计异常处理策略。

```python
class DataValidationError(Exception):
    """数据验证错误（可重试）"""
    def __init__(self, message: str):
        super().__init__(message)
        self.retryable = True

@node(retry_config=RetryConfig(retry_count=3))
def validate_and_process(data: dict) -> dict:
    if not data:
        raise ValueError("Data cannot be empty")  # 不可重试

    if "required_field" not in data:
        raise DataValidationError("Missing required field")  # 可重试

    return {"processed": data}
```

### 性能优化

#### 1. 选择合适的执行器

```python
# I/O密集型任务使用异步执行器
io_flow = fetch_data.fan_out_to(
    targets=[fetch_user, fetch_orders, fetch_products],
    executor="async"
)

# CPU密集型任务使用线程执行器
cpu_flow = process_data.fan_out_to(
    targets=[compute_stats, generate_report, compress_data],
    executor="thread"
)

# 不确定时使用自动选择
auto_flow = mixed_task.fan_out_to(
    targets=[task1, task2, task3],
    executor="auto"
)
```

#### 2. 批处理优化

```python
@node
def batch_process_users(user_ids: List[int], batch_size: int = 100) -> List[dict]:
    """批量处理用户数据"""
    results = []
    for i in range(0, len(user_ids), batch_size):
        batch = user_ids[i:i + batch_size]
        batch_results = process_user_batch(batch)
        results.extend(batch_results)
    return results
```

### 调试技巧

#### 1. 节点命名

```python
@node(name="user_data_extractor")
def extract_user_data(user_id: int) -> dict:
    return get_user_from_db(user_id)
```

#### 2. 日志记录

```python
import logging

logger = logging.getLogger(__name__)

@node
def logged_processor(data: dict) -> dict:
    logger.info(f"Processing data: {data.get('id')}")
    try:
        result = process_data(data)
        logger.info(f"Successfully processed: {data.get('id')}")
        return result
    except Exception as e:
        logger.error(f"Failed to process {data.get('id')}: {e}")
        raise
```

#### 3. 常见问题

**异步/同步混合执行问题**:
```python
# ❌ 错误：在事件循环中创建新循环
def sync_wrapper():
    return asyncio.run(async_function())

# ✅ 正确：让AetherFlow自动处理
flow = sync_node.then(async_node)
```

**重试机制不生效**:
```python
# 确保异常类型正确配置
retry_config = RetryConfig(
    retry_count=3,
    exception_types=(ConnectionError, TimeoutError, YourCustomError)
)

# 或标记自定义异常为可重试
class YourCustomError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.retryable = True  # 重要：标记为可重试
```

---

## API速查

### @node 装饰器

```python
@node(
    name: Optional[str] = None,
    retry_config: Optional[RetryConfig] = None
)
def function_name(...):
    pass
```

- `name`: 节点名称（可选）
- `retry_config`: 重试配置对象

### Node 类方法

#### `then(next_node: Node) -> Node`
顺序连接节点。

#### `fan_out_to(targets: List[Node], executor: str = "async") -> Node`
并行扇出到多个目标节点。

- `targets`: 目标节点列表
- `executor`: 执行器类型（"async", "thread", "auto"）

#### `fan_in(aggregator: Node) -> Node`
聚合并行结果。

- `aggregator`: 聚合器节点，接收 `ParallelResult` 参数

#### `branch_on(conditions: Dict[Any, Node]) -> Node`
条件分支执行。

- `conditions`: 条件映射字典

#### `repeat(times: int, stop_on_error: bool = False) -> Node`
重复执行节点。

- `times`: 重复次数
- `stop_on_error`: 遇到错误时是否停止

### RetryConfig 类

```python
RetryConfig(
    retry_count: int = 3,
    retry_delay: float = 1.0,
    exception_types: Tuple[Type[Exception], ...] = (Exception,),
    backoff_factor: float = 2.0,
    max_delay: float = 60.0
)
```

- `retry_count`: 最大重试次数
- `retry_delay`: 初始重试延迟（秒）
- `exception_types`: 可重试的异常类型元组
- `backoff_factor`: 指数退避因子
- `max_delay`: 最大延迟时间（秒）

### BaseFlowContext 类

流执行上下文，提供依赖注入容器。

```python
container = BaseFlowContext()
container.wire(modules=[__name__])
```

### ParallelResult 类

并行执行结果的容器类，类似字典接口。

```python
result = ParallelResult({"task1": value1, "task2": value2})
print(result["task1"])  # 访问特定任务结果
print(list(result.values()))  # 获取所有结果值
```

### 异常类

#### AetherFlowException
框架基础异常类。

```python
class AetherFlowException(Exception):
    def __init__(self, message: str, node_name: str = None, context: dict = None):
        self.message = message
        self.node_name = node_name
        self.context = context or {}
        self.retryable = False  # 默认不可重试
```

---

**本文档涵盖了AetherFlow的完整功能和最佳实践。如需更多信息，请参考源代码或提交Issue。**
