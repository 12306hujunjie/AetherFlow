# AetherFlow 并发安全使用指南

## 概述

AetherFlow框架通过使用`dependency_injector`库的`ThreadLocalSingleton`提供者，实现了线程安全和协程安全的依赖注入系统。本指南详细说明如何在并发环境中正确使用框架。

## 核心特性

### 1. 线程本地状态隔离

每个线程拥有独立的状态空间，确保并发执行时状态不会互相干扰。

```python
from src.aetherflow import node, AppContext

@node
def worker_function(x):
    # 每个线程都有独立的状态
    return {'thread_result': x + threading.current_thread().ident}
```

### 2. 并行扇出执行

支持将单个输入扇出到多个并行处理节点，并安全地收集结果。

```python
@node
def source(data):
    return data

@node  
def processor_a(source):
    return source + 100

@node
def processor_b(source):
    return source * 2

# 并行扇出
parallel_flow = source.fan_out_to([processor_a, processor_b])
result = parallel_flow.run({'data': 10})
print(result['__parallel_results'])  # {'processor_a': 110, 'processor_b': 20}
```

### 3. 依赖注入容器线程安全

使用`ThreadLocalSingleton`确保每个线程有独立的容器状态。

```python
class BaseFlowContext(containers.DeclarativeContainer):
    """线程安全的依赖注入容器"""
    # 每个线程获得独立的状态字典
    state = providers.ThreadLocalSingleton(dict)
    context = providers.Self()
```

## 使用示例

### 基本并发执行

```python
import threading
from src.aetherflow import node

@node
def increment(x):
    return x + 1

@node 
def multiply(increment, factor):
    return increment * factor

# 创建工作流程
workflow = increment.then(multiply)

def worker(worker_id):
    result = workflow.run({'x': worker_id * 10, 'factor': 2})
    print(f"Worker {worker_id}: {result}")

# 启动多个线程
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

### 高性能并行处理

```python
import concurrent.futures
from src.aetherflow import node

@node
def cpu_intensive_task(data):
    # CPU密集型任务
    result = sum(i * data for i in range(10000))
    return {'computed': result}

# 使用线程池
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for i in range(20):
        future = executor.submit(cpu_intensive_task.run, {'data': i})
        futures.append(future)
    
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
```

## 最佳实践

### 1. 状态管理

- ✅ 使用ThreadLocalSingleton进行状态隔离
- ✅ 避免在节点间共享可变状态
- ❌ 不要使用全局变量存储状态

### 2. 节点设计

```python
# ✅ 好的设计：纯函数式节点
@node
def good_node(x, y):
    return {'result': x + y}

# ❌ 避免：依赖全局状态
global_state = {}

@node 
def bad_node(x):
    global_state['value'] = x  # 不线程安全
    return x
```

### 3. 错误处理

```python
@node
def safe_operation(data):
    try:
        result = risky_computation(data)
        return {'success': True, 'result': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}
```

### 4. 资源管理

```python
import contextlib

@node
def resource_safe_node(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return {'content': content}
    except FileNotFoundError:
        return {'content': None, 'error': 'File not found'}
```

## 性能优化建议

### 1. 选择合适的执行器

```python
# CPU密集型任务
cpu_flow = source.fan_out_to([task1, task2], executor='process')

# I/O密集型任务  
io_flow = source.fan_out_to([task1, task2], executor='thread')
```

### 2. 控制并发度

```python
# 限制最大工作线程数
flow = source.fan_out_to([task1, task2, task3], max_workers=2)
```

### 3. 批处理优化

```python
@node
def batch_processor(items):
    # 批量处理多个项目
    results = []
    for item in items:
        results.append(process_item(item))
    return {'batch_results': results}
```

## 测试并发安全性

框架提供了完整的并发测试套件，位于 `tests/test_concurrent_safety.py`：

```bash
# 运行并发安全测试
python tests/test_concurrent_safety.py
```

测试包括：
- 线程本地状态隔离测试
- 并行扇出线程安全性测试  
- 并发容器访问测试
- 高并发压力测试
- 异步兼容性测试

## 故障排除

### 常见问题

1. **状态混乱**：确保使用ThreadLocalSingleton而非普通Object提供者
2. **死锁**：避免在节点中使用阻塞操作，使用超时机制
3. **内存泄漏**：及时清理线程本地存储，特别是在长运行应用中

### 调试技巧

```python
@node
def debug_node(x):
    import threading
    thread_id = threading.current_thread().ident
    print(f"Thread {thread_id}: processing {x}")
    return {'debug_info': f'processed by {thread_id}'}
```

## 与异步代码集成

虽然框架本身是同步的，但可以在异步环境中使用：

```python
import asyncio

async def async_wrapper():
    loop = asyncio.get_event_loop()
    
    # 在异步环境中运行同步节点
    result = await loop.run_in_executor(
        None,
        lambda: my_node.run({'data': 42})
    )
    
    return result
```

## 小结

AetherFlow框架的并发安全特性让您可以：

- 安全地在多线程环境中执行工作流程
- 利用并行处理提高性能
- 保持代码简洁和可维护性
- 无需担心状态同步问题

遵循本指南的最佳实践，您可以充分利用现代多核处理器的优势，构建高性能的数据处理流水线。