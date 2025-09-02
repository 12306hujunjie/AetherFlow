# AetherFlow - 智能流式数据处理框架

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#测试)

**AetherFlow 通过智能抽象让数据流处理回归业务本质：专注逻辑表达，框架处理执行细节。**

- 🎯 **声明式工作流**：用方法链表达数据处理逻辑，代码即业务流程
- 🤖 **智能执行引擎**：自动处理async/sync混合执行，无需手动协调
- 🔗 **函数式组合**：@node装饰器让任何函数变成可组合节点
- 🛡️ **企业级特性**：内置重试机制、依赖注入、基于pydantic的类型验证
- 📊 **结果跟踪**：ParallelResult类提供详细的执行状态和错误信息

## 🚀 快速开始

```bash
pip install aetherflow  # 或 pdm add aetherflow
```

```python
from aetherflow import node

@node
def double(x: int) -> int: return x * 2

@node
def add_ten(x: int) -> int: return x + 10

# 链式数据流处理 - 框架核心价值
result = double.then(add_ten)(5)  # 20
```

## 📖 核心概念

- **@node装饰器** - 将函数转为可组合的数据处理单元
- **链式调用** - `.then()`, `.fan_out_to()`, `.fan_in()`, `.branch_on()` 构建数据流
- **智能执行** - 自动处理同步/异步函数混合，无需手动协调
- **结果跟踪** - `ParallelResult` 提供并行执行的详细状态信息

| 方法 | 功能 | 用法示例 |
|------|------|----------|
| `.then(node)` | 顺序连接节点 | `a.then(b)(data)` |
| `.fan_out_to([nodes])` | 并行分发到多个节点 | `a.fan_out_to([b, c])()` |
| `.fan_in(aggregator)` | 聚合并行结果 | `flow.fan_in(merge_func)()` |
| `.fan_out_in([nodes], agg)` | 扇出后立即聚合 | `a.fan_out_in([b, c], merge)()` |
| `.branch_on({key: node})` | 条件分支执行 | `a.branch_on({"high": b, "low": c})()` |
| `.repeat(times)` | 重复执行节点 | `a.repeat(3)(data)` |

**ParallelResult结构**: `{'node_name': ParallelResult(success=bool, result=value, execution_time=float)}`

## 🎯 完整示例

```python
from aetherflow import node
import asyncio

# 定义处理节点
@node
async def fetch_data(source: str) -> list:
    await asyncio.sleep(0.1)  # 模拟异步IO
    return [{"id": 1, "score": 85}, {"id": 2, "score": 45}]

@node
def validate_item(item: dict) -> dict:
    return {**item, "valid": item["score"] > 0}

@node
def enrich_item(item: dict) -> dict:
    return {**item, "grade": "A" if item["score"] >= 60 else "F"}

@node
def merge_results(parallel_results: dict) -> list:
    return [r.result for r in parallel_results.values() if r.success]

@node
def classify_by_grade(items: list) -> str:
    avg_score = sum(item["score"] for item in items) / len(items)
    return "high" if avg_score >= 60 else "low"

@node
def generate_report(items: list) -> dict:
    return {"report": "success", "processed": len(items)}

@node
def send_alert(items: list) -> dict:
    return {"alert": "sent", "low_score_count": len(items)}

# 构建完整数据流：异步提取 → 并行处理 → 聚合 → 条件分支
workflow = (
    fetch_data
    .fan_out_to([validate_item, enrich_item])  # 并行处理每条数据
    .fan_in(merge_results)                     # 聚合结果
    .then(classify_by_grade)                   # 分类评估
    .branch_on({                               # 条件分支
        "high": generate_report,
        "low": send_alert
    })
)

# 执行
async def main():
    result = await workflow("database")
    print(result)  # {"alert": "sent", "low_score_count": 2}

asyncio.run(main())
```

## 📊 对比优势

| 特性 | AetherFlow | Celery | Airflow | Ray |
|------|-----------|--------|---------|-----|
| **学习成本** | 低（函数式） | 中 | 高 | 中 |
| **async/sync混合** | ✅ 自动处理 | ❌ | ❌ | ✅ 手动 |
| **声明式设计** | ✅ 方法链 | ❌ | ❌ | ✅ |
| **轻量部署** | ✅ 最小依赖 | ❌ 需Redis | ❌ 需集群 | ❌ |

> **适用场景**：中小型数据处理任务，ai agents flow 构建，微服务数据管道，强调代码可读性和快速开发

## 📚 文档资源

- 📖 **完整文档**：[AetherFlow技术文档.md](docs/AetherFlow技术文档.md) - 深入了解架构设计和最佳实践
- 💡 **代码示例**：`tests/` 目录 - 实际使用场景和测试用例
- 🛠️ **开发参考**：`CLAUDE.md` - Claude Code集成开发指南
- 🔍 **API参考**：技术文档中的完整API说明

## 🔧 开发环境

### 开发者安装
```bash
# 克隆项目
git clone https://github.com/12306hujunjie/AetherFlow.git
cd AetherFlow

# 使用PDM管理开发环境
pdm install  # 安装所有依赖（包括开发依赖）

# 运行测试
pdm run pytest

# 代码质量检查
pdm run lint      # 代码检查
pdm run format    # 代码格式化
pdm run type-check # 类型检查
```

### 开发工具
```bash
# 完整测试套件
pdm run test-all     # 运行所有测试
pdm run test-cov     # 带覆盖率的测试

# 文档测试
pdm run doc-test     # 文档示例测试
pdm run doc-test-verbose  # 详细输出

# 预提交检查
pdm run pre-commit   # 运行所有质量检查
```

## 🔧 技术栈

**运行时依赖**：
- **Python 3.10+** - 现代Python特性支持
- **dependency-injector ≥ 4.48.1** - 依赖注入和状态管理
- **pydantic ≥ 2.11.7** - 数据验证和类型安全
- **httpx ≥ 0.28.1** - 异步HTTP客户端支持

**核心模块**：`asyncio` | `threading` | `concurrent.futures`

## 📄 许可证

MIT许可证 - 详见 [LICENSE](LICENSE)

---
**立即开始构建您的智能数据处理系统！** 🚀
