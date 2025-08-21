# AetherFlow 架构决策指南

## 核心发现：双并发模式支持

通过深度分析，AetherFlow框架实际支持**两种截然不同的并发模式**，每种模式适用于不同的业务场景。

## 🎯 并发模式对比

### 模式1：线程隔离模式 (Thread Isolation)

**技术实现**：
```python
class IsolatedContext(BaseFlowContext):
    """线程隔离上下文 - 每个线程独立状态"""
    state = providers.ThreadLocalSingleton(dict)
    service = providers.ThreadLocalSingleton(MyService)
```

**特征**：
- ✅ 每个线程拥有完全独立的状态空间
- ✅ 天然线程安全，无需锁机制  
- ✅ 函数式编程范式，易于推理
- ✅ 无竞争条件，调试简单
- ❌ 内存使用较高（每线程一份状态）
- ❌ 无法进行线程间通信

**适用场景**：
```python
# ✅ 完美适用：独立任务处理
@node
def process_user_data(user_id, state):
    # 每个用户数据处理完全独立
    return {'processed_user': user_id}

# ✅ 完美适用：MapReduce型并行处理  
@node
def map_task(data_chunk, state):
    return {'chunk_result': process_chunk(data_chunk)}

# ✅ 完美适用：无状态服务调用
@node  
def api_call(endpoint, http_client):
    return http_client.get(endpoint)
```

### 模式2：共享状态模式 (Shared State)

**技术实现**：
```python
class SharedStateService:
    def __init__(self):
        self.counter = 0
        self.results = []
        self.lock = threading.Lock()  # 手动同步
    
    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter

class SharedContext(BaseFlowContext):
    """共享状态上下文 - 线程间共享状态"""
    shared_service = providers.Singleton(SharedStateService)
```

**特征**：
- ✅ 线程间可以协调和通信
- ✅ 内存使用效率高（单例共享）
- ✅ 支持复杂的业务协调逻辑
- ✅ 传统OOP模式，易于理解
- ❌ 需要手动同步机制（锁、信号量等）
- ❌ 潜在竞争条件和死锁风险
- ❌ 调试复杂度高

**适用场景**：
```python
# ✅ 完美适用：需要全局协调的任务
@node
def coordinated_worker(task_id, coordinator):
    # 需要与其他线程协调的工作
    return coordinator.claim_next_task(task_id)

# ✅ 完美适用：共享资源池管理
@node
def database_operation(query, connection_pool):
    with connection_pool.get_connection() as conn:
        return conn.execute(query)

# ✅ 完美适用：实时状态聚合
@node
def update_dashboard(metrics, dashboard_service):
    dashboard_service.update_real_time_metrics(metrics)
    return {'updated': True}
```

## 🚀 架构决策框架

### 决策树

```
业务需求分析
├─ 是否需要线程间通信？
│  ├─ 是 → 共享状态模式
│  └─ 否 → 继续判断
├─ 是否是独立任务处理？
│  ├─ 是 → 线程隔离模式  
│  └─ 否 → 继续判断
├─ 团队并发编程经验如何？
│  ├─ 丰富 → 可选择共享状态模式
│  └─ 有限 → 推荐线程隔离模式
└─ 性能要求如何？
   ├─ 内存敏感 → 共享状态模式
   └─ 安全优先 → 线程隔离模式
```

### 场景对应表

| 业务场景 | 推荐模式 | 理由 |
|---------|---------|------|
| 数据ETL处理 | 线程隔离 | 独立处理，无需协调 |
| 实时计数器 | 共享状态 | 需要全局状态聚合 |
| 文件批处理 | 线程隔离 | 每个文件独立处理 |
| 连接池管理 | 共享状态 | 资源需要共享 |
| 用户请求处理 | 线程隔离 | 用户间互相独立 |
| 系统监控 | 共享状态 | 需要全局视图 |
| 机器学习推理 | 线程隔离 | 每个样本独立 |
| 分布式锁 | 共享状态 | 需要全局协调 |

## ⚖️ 性能与安全权衡

### 内存使用对比

```python
# 线程隔离模式：n个线程 = n份状态
# 4线程 × 1MB状态 = 4MB内存使用

# 共享状态模式：n个线程 = 1份状态  
# 4线程 × 1MB状态 = 1MB内存使用 + 同步开销
```

### CPU开销对比

```python
# 线程隔离：无锁开销，但有线程本地存储访问成本
# 共享状态：锁竞争开销，可能的上下文切换

# 基准测试结果（见performance_benchmarks.py）：
# - 轻量级操作：线程隔离胜出 ~15%
# - 重量级操作：差异不明显 ~3%  
# - 高竞争场景：线程隔离胜出 ~40%
```

## 🔄 模式迁移指南

### 从共享状态到线程隔离

```python
# 迁移前：共享状态
class OldContext(BaseFlowContext):
    shared_service = providers.Singleton(MyService)

# 迁移后：线程隔离
class NewContext(BaseFlowContext):
    isolated_service = providers.ThreadLocalSingleton(MyService)

# 节点代码无需修改！
@node
def my_node(isolated_service: MyService):
    # 行为改变：每线程独立实例
    return isolated_service.process()
```

### 混合模式使用

```python
class HybridContext(BaseFlowContext):
    """混合模式：根据需要选择不同provider"""
    
    # 线程隔离的用户状态
    user_state = providers.ThreadLocalSingleton(dict)
    
    # 共享的连接池
    db_pool = providers.Singleton(ConnectionPool)
    
    # 共享的缓存服务
    cache = providers.Singleton(CacheService)

@node
def hybrid_operation(user_state: dict, db_pool: ConnectionPool):
    """混合使用：用户状态隔离 + 连接池共享"""
    with db_pool.get_connection() as conn:
        # 用户状态是线程独立的
        user_state['last_query'] = time.time()
        return conn.query("SELECT * FROM users")
```

## 🎨 最佳实践模式

### 模式1：纯函数式流水线
```python
class PureFunctionalContext(BaseFlowContext):
    """纯函数式上下文"""
    state = providers.ThreadLocalSingleton(dict)

@node
def pure_transform(data, state):
    # 无副作用的数据转换
    result = transform_data(data)
    state['processed_count'] = state.get('processed_count', 0) + 1
    return result
```

### 模式2：协调器模式
```python
class CoordinatorContext(BaseFlowContext):
    """协调器模式上下文"""
    coordinator = providers.Singleton(TaskCoordinator)

@node
def coordinated_task(task_data, coordinator):
    # 通过协调器分配任务
    assigned_task = coordinator.assign_task()
    result = process_task(assigned_task, task_data)
    coordinator.complete_task(assigned_task, result)
    return result
```

### 模式3：资源池模式
```python
class ResourcePoolContext(BaseFlowContext):
    """资源池模式上下文"""
    http_pool = providers.Singleton(HTTPConnectionPool)
    db_pool = providers.Singleton(DatabasePool)

@node
def resource_task(request, http_pool, db_pool):
    # 使用资源池处理请求
    with http_pool.get_session() as session:
        api_data = session.get(request['url'])
    
    with db_pool.get_connection() as db:
        db.save(api_data)
    
    return {'saved': True}
```

## 🔍 调试和监控

### 线程隔离模式调试
```python
@node
def debug_isolated(data, state):
    import threading
    thread_id = threading.current_thread().ident
    state[f'debug_{thread_id}'] = {
        'data': data,
        'timestamp': time.time()
    }
    print(f"Thread {thread_id}: {state}")
    return data
```

### 共享状态模式监控
```python
class MonitoredSharedService:
    def __init__(self):
        self.operations = 0
        self.lock = threading.Lock()
        self.metrics = defaultdict(int)
    
    def track_operation(self, op_type):
        with self.lock:
            self.operations += 1
            self.metrics[op_type] += 1
            
            if self.operations % 100 == 0:
                print(f"Operations: {self.operations}, Metrics: {dict(self.metrics)}")
```

## 📊 选择矩阵

| 考虑因素 | 线程隔离 | 共享状态 | 权重 |
|---------|---------|---------|------|
| 开发简单性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 高 |
| 运行时安全性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 高 |
| 内存效率 | ⭐⭐ | ⭐⭐⭐⭐⭐ | 中 |
| 协调能力 | ⭐ | ⭐⭐⭐⭐⭐ | 中 |
| 调试友好性 | ⭐⭐⭐⭐⭐ | ⭐⭐ | 高 |
| 扩展性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中 |

## 💡 架构演进路径

### 阶段1：新手友好（推荐起点）
```python
# 默认选择线程隔离模式
class BeginnerContext(BaseFlowContext):
    state = providers.ThreadLocalSingleton(dict)
```

### 阶段2：混合使用
```python  
# 根据具体需求混合使用
class IntermediateContext(BaseFlowContext):
    user_state = providers.ThreadLocalSingleton(dict)
    shared_cache = providers.Singleton(CacheService)
```

### 阶段3：高级优化
```python
# 根据性能特征定制
class AdvancedContext(BaseFlowContext):
    # 基于基准测试结果选择最优provider
    optimized_service = providers.ThreadLocalSingleton(OptimizedService)
```

## 🎯 结论

AetherFlow的双并发模式设计体现了**"灵活性与安全性并重"**的哲学：

- **默认安全**：推荐新用户使用线程隔离模式
- **按需优化**：高级用户可根据场景选择共享状态模式  
- **渐进学习**：支持从简单到复杂的学习路径
- **向后兼容**：现有代码无需修改即可享受改进

选择合适的模式不是技术问题，而是**业务架构决策**。理解你的数据流、协调需求和团队能力，然后做出明智的选择。