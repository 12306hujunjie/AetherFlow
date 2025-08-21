# AetherFlow 快速入门指南

## 什么是 AetherFlow？

AetherFlow 是一个现代化的 Python 数据流处理框架，专为构建可扩展、可维护的数据处理管道而设计。它采用**节点式编程模型**，让您可以将复杂的业务逻辑分解为简单、可重用的处理单元。

## 🌟 核心特性

- **🔗 声明式流程定义**: 通过链式API构建清晰的数据流
- **🧵 内置并发支持**: 支持线程隔离和共享状态两种并发模式
- **💉 依赖注入系统**: 基于dependency_injector的强大DI支持
- **🚀 高性能执行**: 优化的并行执行和资源管理
- **🛡️ 线程安全**: 默认提供线程安全的状态管理
- **🎯 类型提示友好**: 完整的类型注解支持

## 🚀 快速开始

### 安装

```bash
pip install aetherflow
```

### 第一个示例

```python
from aetherflow import node

# 定义处理节点
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
    return {'filtered_data': filtered, 'original_count': load_data['count'], 'filtered_count': len(filtered)}

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

# 构建数据处理管道
pipeline = (load_data
            .then(filter_data)
            .then(save_results))

# 执行管道
result = pipeline.run({
    'filename': 'input.txt',
    'min_length': 3,
    'output_file': 'output.txt'
})

print(f"处理完成: {result}")
```

## 📋 基本概念

### 1. 节点 (Node)

节点是AetherFlow的基本处理单元，使用`@node`装饰器定义：

```python
@node
def my_processor(input_data, param1, param2=None):
    """
    节点函数说明：
    - 第一个参数通常是输入数据
    - 可以有多个参数
    - 支持默认值
    - 返回字典作为输出
    """
    result = process(input_data, param1, param2)
    return {'processed': result}
```

### 2. 数据流 (Data Flow)

节点可以通过多种方式连接形成数据流：

```python
# 顺序执行
flow1 = node_a.then(node_b).then(node_c)

# 并行扇出
flow2 = source.fan_out_to([processor_a, processor_b, processor_c])

# 扇入聚合
flow3 = source.fan_out_to([proc_a, proc_b]).fan_in(aggregator)

# 条件分支
flow4 = source.branch({
    'condition_a': processor_a,
    'condition_b': processor_b,
    'default': default_processor
})
```

### 3. 上下文 (Context)

上下文提供依赖注入和状态管理：

```python
from aetherflow import BaseFlowContext
from dependency_injector import providers

class MyContext(BaseFlowContext):
    """自定义上下文"""
    # 线程本地状态 (推荐)
    state = providers.ThreadLocalSingleton(dict)
    
    # 自定义服务
    database = providers.ThreadLocalSingleton(DatabaseService)
    cache = providers.ThreadLocalSingleton(CacheService)

@node
def data_processor(input_data, database: DatabaseService, cache: CacheService):
    """使用注入的服务"""
    # 检查缓存
    if cache.has(input_data['key']):
        return cache.get(input_data['key'])
    
    # 从数据库获取
    result = database.query(input_data['query'])
    cache.set(input_data['key'], result)
    
    return result

# 使用自定义上下文
context = MyContext()
result = data_processor.run({'key': 'user:123', 'query': 'SELECT * FROM users'}, context)
```

## 🛡️ 并发模式选择

AetherFlow支持两种并发模式，根据您的需求选择：

### 模式1：线程隔离 (推荐新手)

```python
class IsolatedContext(BaseFlowContext):
    """线程隔离上下文 - 每个线程独立状态"""
    state = providers.ThreadLocalSingleton(dict)
    service = providers.ThreadLocalSingleton(MyService)

# 特点：
# ✅ 线程安全，无竞争条件
# ✅ 调试简单
# ✅ 适合独立任务处理
# ❌ 内存使用较高
```

### 模式2：共享状态 (适合高级用户)

```python
class SharedStateService:
    def __init__(self):
        self.counter = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter

class SharedContext(BaseFlowContext):
    """共享状态上下文 - 需要手动同步"""
    shared_service = providers.Singleton(SharedStateService)

# 特点：
# ✅ 内存效率高
# ✅ 支持线程间协调
# ❌ 需要并发编程经验
# ❌ 潜在竞争条件
```

**如何选择？**
- 🎯 **独立任务处理** → 选择线程隔离模式
- 🎯 **需要线程协调** → 选择共享状态模式
- 🎯 **新手入门** → 推荐线程隔离模式
- 🎯 **内存敏感** → 考虑共享状态模式

## 💡 实用模式

### 1. ETL 数据处理

```python
@node
def extract(source_config):
    """提取数据"""
    data = load_from_source(source_config)
    return {'raw_data': data, 'record_count': len(data)}

@node  
def transform(extract, transformation_rules):
    """转换数据"""
    transformed = apply_transformations(extract['raw_data'], transformation_rules)
    return {'transformed_data': transformed}

@node
def load(transform, target_config):
    """加载数据"""
    save_to_target(transform['transformed_data'], target_config)
    return {'loaded_records': len(transform['transformed_data']), 'success': True}

# ETL管道
etl_pipeline = extract.then(transform).then(load)
```

### 2. 批处理任务

```python
@node
def batch_processor(data_batch, batch_size=100):
    """批处理数据"""
    results = []
    for i in range(0, len(data_batch['items']), batch_size):
        batch = data_batch['items'][i:i + batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
    
    return {'results': results, 'total_processed': len(results)}
```

### 3. 并行处理

```python
@node
def data_source(input_config):
    """数据源"""
    return {'data_chunks': split_data(input_config)}

@node
def parallel_processor_a(data_source):
    """并行处理器A"""
    return {'result_a': process_type_a(data_source['data_chunks'])}

@node
def parallel_processor_b(data_source):
    """并行处理器B"""  
    return {'result_b': process_type_b(data_source['data_chunks'])}

@node
def combine_results(parallel_results):
    """合并并行结果"""
    combined = merge_results([
        parallel_results['parallel_processor_a']['result_a'],
        parallel_results['parallel_processor_b']['result_b']
    ])
    return {'final_result': combined}

# 并行管道
parallel_flow = (data_source
                 .fan_out_to([parallel_processor_a, parallel_processor_b])
                 .fan_in(combine_results))
```

## 🔧 调试和监控

### 添加调试信息

```python
@node
def debug_node(input_data):
    """带调试的节点"""
    print(f"🔍 处理数据: {input_data}")
    
    result = process_data(input_data)
    
    print(f"✅ 处理完成: {len(result)} 条记录")
    return result
```

### 错误处理

```python
@node
def safe_processor(input_data):
    """安全的处理器"""
    try:
        result = risky_operation(input_data)
        return {'success': True, 'result': result}
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'input_data': input_data
        }

@node
def error_handler(safe_processor):
    """错误处理"""
    if not safe_processor['success']:
        # 记录错误并使用默认值
        log_error(safe_processor['error'])
        return {'result': get_default_result(), 'recovered': True}
    
    return safe_processor['result']
```

## 🎯 最佳实践

### 1. 节点设计原则

```python
# ✅ 好的设计：单一职责、纯函数
@node
def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return {'is_valid': bool(re.match(pattern, email)), 'email': email}

# ❌ 避免：多重职责、副作用
@node  
def validate_and_send_email(email, message):
    # 同时验证和发送，违反单一职责原则
    pass
```

### 2. 类型提示

```python
from typing import Dict, Any, List

@node
def typed_processor(
    data: Dict[str, Any], 
    threshold: float = 0.5
) -> Dict[str, Any]:
    """类型化的处理器"""
    filtered_items = [
        item for item in data['items'] 
        if item['score'] >= threshold
    ]
    
    return {
        'filtered_items': filtered_items,
        'original_count': len(data['items']),
        'filtered_count': len(filtered_items)
    }
```

### 3. 配置管理

```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    batch_size: int = int(os.getenv('BATCH_SIZE', '100'))
    max_retries: int = int(os.getenv('MAX_RETRIES', '3'))

config = Config()

@node
def configured_processor(data, config: Config = config):
    """使用配置的处理器"""
    return process_with_config(data, config)
```

## 🚀 下一步

现在您已经掌握了AetherFlow的基础用法！建议您：

1. **阅读架构决策指南** (`docs/architecture_guide.md`) - 了解如何选择合适的并发模式
2. **查看最佳实践** (`docs/best_practices.md`) - 学习高级使用模式
3. **运行性能基准** (`tests/performance_benchmarks.py`) - 了解性能特征
4. **查看示例代码** (`examples/`) - 学习实际应用案例

## 🤝 获取帮助

- **文档**: 查看 `docs/` 目录下的详细文档
- **示例**: 查看 `examples/` 目录下的实际案例
- **测试**: 查看 `tests/` 目录下的测试用例

开始构建您的第一个AetherFlow应用吧！ 🚀