# AetherFlow - 智能流式接口框架

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/12306hujunjie/AetherFlow/actions/workflows/test.yml/badge.svg)](https://github.com/12306hujunjie/AetherFlow/actions/workflows/test.yml)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#测试)
[![Coverage](https://img.shields.io/badge/coverage-90%25+-brightgreen.svg)](#测试覆盖率)

一个现代化的 Python 数据流处理框架，专为构建可扩展、线程安全的数据处理管道而设计。支持双并发模式、智能依赖注入和企业级性能优化。

## 🌟 核心特性

- **🔗 声明式流程定义**: 通过链式API构建清晰的数据流
- **🧵 双并发模式**: 支持线程隔离和共享状态两种并发策略
- **💉 智能依赖注入**: 基于dependency_injector的线程安全DI系统
- **🚀 高性能执行**: 经测试在高并发下性能提升达38%
- **🛡️ 线程安全保证**: 默认ThreadLocalSingleton确保并发安全
- **⚡ 并行处理能力**: 扇出/扇入模式支持复杂的并行工作流
- **🎯 类型提示友好**: 完整的类型注解和IDE支持
- **📊 生产就绪**: 包含性能监控、错误处理和最佳实践

## 🚀 快速开始

### 安装依赖

```bash
pip install dependency-injector
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

## 🏗️ 并发模式选择

AetherFlow支持两种并发模式，根据业务需求灵活选择：

### 模式1：线程隔离 (推荐)

```python
from aetherflow import BaseFlowContext
from dependency_injector import providers

class IsolatedContext(BaseFlowContext):
    """线程隔离上下文 - 每个线程独立状态"""
    state = providers.ThreadLocalSingleton(dict)
    service = providers.ThreadLocalSingleton(MyService)

# 特点：
# ✅ 天然线程安全，无竞争条件
# ✅ 调试简单，状态清晰
# ✅ 高并发性能优异 (+38% @ 16线程)
# ✅ 适合独立任务处理
```

### 模式2：共享状态 (高级)

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
    """共享状态上下文 - 线程间协调"""
    shared_service = providers.Singleton(SharedStateService)

# 特点：
# ✅ 内存使用效率高
# ✅ 支持线程间协调
# ❌ 需要并发编程经验
# ❌ 需要手动同步
```

**选择指导**：
- 🎯 **独立任务处理** → 线程隔离模式
- 🎯 **需要线程协调** → 共享状态模式
- 🎯 **新手开发者** → 线程隔离模式
- 🎯 **高并发场景** → 线程隔离模式

## ⚡ 并行处理

### 扇出/扇入模式

```python
@node
def data_source():
    return {'data': list(range(1000))}

@node
def processor_a(data_source):
    """处理器A"""
    return {'result_a': sum(data_source['data'])}

@node
def processor_b(data_source):
    """处理器B"""
    return {'result_b': len(data_source['data'])}

@node
def combine_results(parallel_results):
    """合并结果"""
    return {
        'sum': parallel_results['processor_a']['result_a'],
        'count': parallel_results['processor_b']['result_b'],
        'average': parallel_results['processor_a']['result_a'] / parallel_results['processor_b']['result_b']
    }

# 并行管道：源数据 → 并行处理 → 结果合并
pipeline = (data_source
            .fan_out_to([processor_a, processor_b])
            .fan_in(combine_results))

result = pipeline.run({})
print(f"平均值: {result['average']}")
```

### 高性能并发处理

```python
import concurrent.futures
from aetherflow import node

@node
def cpu_intensive_task(data):
    """CPU密集型任务"""
    result = sum(i * data for i in range(10000))
    return {'computed': result}

# 使用线程池处理大量任务
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for i in range(100):
        future = executor.submit(cpu_intensive_task.run, {'data': i})
        futures.append(future)
    
    results = [f.result() for f in concurrent.futures.as_completed(futures)]
    print(f"处理了 {len(results)} 个任务")
```

## 🎯 使用场景

### ETL 数据处理

```python
# 提取 → 转换 → 加载
etl_pipeline = (extract_from_database
                .then(transform_data)
                .then(load_to_warehouse))
```

### 机器学习推理

```python
# 预处理 → 并行推理 → 后处理
ml_pipeline = (preprocess_data
               .fan_out_to([model_a, model_b, model_c])
               .fan_in(ensemble_predictions))
```

### 实时数据处理

```python
# 数据接收 → 并行分析 → 结果聚合
realtime_pipeline = (receive_events
                     .fan_out_to([fraud_detection, sentiment_analysis])
                     .fan_in(generate_alerts))
```

## 📊 性能数据

基于真实基准测试的性能表现：

| 场景 | 线程隔离模式 | 共享状态模式 | 性能提升 |
|------|-------------|-------------|----------|
| 轻量级操作 (8线程) | 45,000 ops/sec | 38,000 ops/sec | **+18%** |
| 重量级操作 (4线程) | 850 ops/sec | 820 ops/sec | **+3.7%** |
| 高并发场景 (16线程) | 52,400 ops/sec | 38,100 ops/sec | **+38%** |

**关键发现**：
- 🚀 线程隔离模式在高并发下优势明显
- 📈 扩展性好，随线程数增加性能优势递增
- ⚡ 消除锁竞争，提供更好的并发性能

## 📚 完整文档

我们提供分层的学习路径，从入门到精通：

### 🚀 入门级 (0-1 小时)
- **[快速入门指南](docs/getting_started.md)** - 20分钟上手基本功能

### 🏗️ 进阶级 (1-4 小时)  
- **[架构决策指南](docs/architecture_guide.md)** - 双并发模式深度解析
- **[并发安全使用指南](docs/concurrent_guide.md)** - 线程安全最佳实践
- **[最佳实践指南](docs/best_practices.md)** - 代码设计和优化策略

### 🏎️ 专家级 (4+ 小时)
- **[性能优化指南](docs/performance_optimization.md)** - 生产环境调优

**完整文档索引**: [docs/README.md](docs/README.md)

## 🧪 测试与验证

### 运行测试

```bash
# 基础功能测试
.venv/bin/python tests/test_concurrent_safety.py

# 复杂服务类测试  
.venv/bin/python tests/test_service_class_safety.py

# 性能基准测试
.venv/bin/python tests/performance_benchmarks.py
```

### 测试覆盖

- ✅ 线程安全性验证
- ✅ 并发状态隔离测试
- ✅ 复杂服务类测试
- ✅ 性能基准对比
- ✅ 高负载压力测试

## 🎨 示例代码

查看 `examples/` 目录获取更多实际使用案例：

- `examples/concurrent_example.py` - 并发处理演示
- `examples/context_demo.py` - 上下文使用示例
- 更多示例持续添加...

## 🔧 开发与贡献

### 项目结构

```
AetherFlow/
├── src/aetherflow/          # 核心框架代码
├── docs/                    # 分层文档系统
├── tests/                   # 完整测试套件
├── examples/                # 实际使用示例
└── README.md                # 项目说明
```

### 技术栈

- **核心**: Python 3.8+
- **依赖注入**: dependency-injector
- **并发**: threading, concurrent.futures
- **类型支持**: typing, dataclasses

## 📈 路线图

- [x] ✅ 线程安全依赖注入系统
- [x] ✅ 双并发模式支持  
- [x] ✅ 性能基准测试体系
- [x] ✅ 完整文档体系
- [ ] 🔄 异步/await 支持
- [ ] 🔄 分布式执行支持
- [ ] 🔄 可视化流程编辑器
- [ ] 🔄 更多集成示例

## 🤝 获取帮助

- **文档**: 查看 [docs/](docs/) 目录
- **示例**: 查看 [examples/](examples/) 目录  
- **测试**: 查看 [tests/](tests/) 目录
- **问题**: 提交 GitHub Issues

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**开始构建您的高性能数据处理系统！** 🚀

```bash
# 克隆项目
git clone <repository-url>
cd AetherFlow

# 设置虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install dependency-injector

# 运行第一个示例
python examples/concurrent_example.py
```