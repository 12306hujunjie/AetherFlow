# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 核心原则

- **理解优于实现**: 深入分析现有代码结构和设计意图后再修改
- **渐进优于激进**: 小步骤改进，每个变更都可验证和回滚
- **证据优于假设**: 所有决策基于实际测试和分析，不依赖猜测
- **简单优于复杂**: 优先选择最简单有效的解决方案

## 项目架构

### 核心概念
AetherFlow是一个声明式数据处理工作流框架，核心架构基于：

- **Node类**: 执行图中的基本单元，支持fluent interface链式调用
- **流式接口**: `.then()`, `.fan_out_to()`, `.fan_in()`, `.branch_on()`, `.repeat()` 方法
- **并行处理**: 基于ThreadPoolExecutor/ProcessPoolExecutor的扇出/扇入模式
- **依赖注入**: 集成dependency-injector的线程安全状态管理
- **重试机制**: 可配置的指数退避重试装饰器

### 关键文件结构
- `src/aetherflow/__init__.py`: 单一模块包含所有核心功能（~1000行）
- `tests/`: 全面的测试套件，使用@node装饰器模式避免pickle问题
- `tests/utils/node_factory.py`: 标准化测试节点定义
- `tests/shared/`: 共享数据模型和测试常量

### 核心类和方法
- `Node.__init__()`: 节点创建，支持重试配置和依赖注入
- `Node.then()`: 顺序链接 (src/aetherflow/__init__.py:322)
- `Node.fan_out_to()`: 并行扇出 (src/aetherflow/__init__.py:326)
- `Node.fan_in()`: 结果汇入 (src/aetherflow/__init__.py:335)
- `Node.branch_on()`: 条件分支 (src/aetherflow/__init__.py:349)
- `Node.repeat()`: 重复执行 (src/aetherflow/__init__.py:353)

## 开发命令

### 环境管理
```bash
# 使用pdm管理虚拟环境（必须）
pdm install               # 安装所有依赖
pdm install --dev        # 安装开发依赖
pdm list                 # 查看依赖树
```

### 测试命令
```bash
# 运行所有测试
pdm run python -m pytest

# 运行特定测试文件
pdm run python -m pytest tests/test_then_core.py -v
pdm run python -m pytest tests/test_fan_primitives.py -v
pdm run python -m pytest tests/test_conditional_composition.py -v

# 运行特定测试函数
pdm run python -m pytest tests/test_fan_primitives.py::test_fan_out_to_executor_types -v

# 带覆盖率报告
pdm run python -m pytest --cov=src/aetherflow --cov-report=html

# 并行测试（如果需要）
pdm run python -m pytest -n auto
```

### 代码质量检查
```bash
# Linting (Ruff配置在pyproject.toml中)
pdm run ruff check src/ tests/
pdm run ruff format src/ tests/

# 类型检查 (MyPy配置在pyproject.toml中)
pdm run mypy src/aetherflow/

# 安全扫描 (Bandit配置排除tests目录)
pdm run bandit -r src/
```

## 技术约束

### 环境要求
- **必须使用PDM**: 不使用pip、conda或全局Python环境
- **Python 3.10+**: 项目需要现代Python特性
- **依赖管理**: 新依赖需要添加到pyproject.toml并评估必要性

### 代码风格
- **Ruff格式化**: 88字符行长，双引号，空格缩进
- **类型注解**: 所有函数必须有类型注解（MyPy strict模式）
- **Pydantic验证**: 使用Pydantic BaseModel进行数据验证
- **日志记录**: 使用标准logging库，logger名称为"aetherflow"

### 架构约束
- **单模块设计**: 核心功能集中在`__init__.py`中，避免过度拆分
- **线程安全**: 使用ThreadLocalSingleton模式进行状态隔离
- **可序列化**: 所有Node必须支持pickle序列化（用于进程池）
- **依赖注入**: 使用dependency-injector容器管理状态
- **组合层次**: composition函数使用Node类，@node装饰器仅用于用户业务节点

## 测试原则

### 测试结构
- **使用@node装饰器**: 测试节点定义在模块级别，支持pickle序列化
- **共享基础设施**: 使用`tests/utils/node_factory.py`的标准化节点
- **数据模型**: 使用`tests/shared/data_models.py`的Pydantic模型
- **隔离性**: 每个测试使用独立的依赖注入容器

### 测试覆盖
- **核心原语测试**: then、fan_out、fan_in、branch_on、repeat功能
- **并发安全测试**: 线程池和进程池执行器的正确性
- **错误处理测试**: 重试机制、异常传播、超时处理
- **依赖注入测试**: 状态管理和线程隔离

### 常用测试模式
```python
# 使用标准节点工厂
from tests.utils.node_factory import add_10_node, double_node

# 构建测试流程
flow = add_10_node.then(double_node)
result = flow(5)  # (5 + 10) * 2 = 30

# 并行测试验证
from tests.utils.parallel_utils import verify_parallel_execution
verify_parallel_execution(flow, input_data, expected_results)
```

## 调试和问题排查

### 常见问题
1. **Pickle序列化错误**: 确保节点函数定义在模块级别，不使用lambda
2. **依赖注入失败**: 检查容器wire配置和Provide注解
3. **并发竞争条件**: 验证ThreadLocalSingleton状态隔离
4. **测试超时**: 检查并行执行器的max_workers配置

### 调试技巧
- 启用详细日志: `logging.getLogger("aetherflow").setLevel(logging.DEBUG)`
- 检查并行结果: 使用`ParallelResult`数据类分析执行状态
- 状态检查: 通过依赖注入访问线程本地状态

## 数据流处理特殊考虑

### 性能优化
- **线程vs进程**: 默认使用ThreadPoolExecutor，CPU密集任务考虑ProcessPoolExecutor
- **状态管理**: 避免大对象在线程间传递，使用引用或标识符
- **内存效率**: 大数据流使用生成器模式，避免全量加载

### 错误恢复
- **分层重试**: Node级别重试 + 组合级别重试
- **异常分类**: 区分瞬时异常(TRANSIENT_EXCEPTIONS)和永久异常(PERMANENT_EXCEPTIONS)
- **状态回滚**: 失败时的状态清理和恢复机制

### 监控和观测
- **执行跟踪**: 记录每个节点的执行时间和状态
- **并行可视化**: 使用ParallelResult分析并发执行效果
- **依赖注入诊断**: 验证容器配置和服务注册状态
