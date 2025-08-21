# AetherFlow 最佳实践指南

## 架构设计原则

### 1. 节点设计原则

#### 单一职责原则
每个节点应该只负责一个明确的功能：

```python
# ✅ 好的设计：单一职责
@node
def validate_email(email):
    """验证邮箱格式"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return {'is_valid': bool(re.match(pattern, email))}

@node
def send_email(email, message):
    """发送邮件"""
    # 发送邮件逻辑
    return {'sent': True, 'timestamp': time.time()}

# ❌ 不好的设计：多重职责
@node
def validate_and_send_email(email, message):
    """既验证又发送邮件"""
    # 违反单一职责原则
    pass
```

#### 纯函数设计
节点应该是纯函数，避免副作用：

```python
# ✅ 纯函数：相同输入总是产生相同输出
@node
def calculate_tax(amount, rate):
    return {'tax': amount * rate}

# ❌ 有副作用：依赖外部状态
import datetime
@node
def calculate_tax_with_date(amount):
    # 依赖当前日期，不是纯函数
    current_date = datetime.now()
    # ...
```

### 2. 依赖注入最佳实践

#### 明确依赖关系
```python
from typing import Dict, Any

@node
def process_user_data(user_id: int, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    明确指定参数类型和返回值类型
    """
    user_info = state.get('user_info', {})
    return {
        'processed_user': {
            'id': user_id,
            'name': user_info.get('name'),
            'processed_at': time.time()
        }
    }
```

#### 容器配置
```python
class CustomFlowContext(BaseFlowContext):
    """自定义流程容器"""
    
    # 配置特定的提供者
    database = providers.ThreadLocalSingleton(DatabaseConnection)
    cache = providers.ThreadLocalSingleton(CacheService)
    logger = providers.ThreadLocalSingleton(Logger)
```

### 3. 错误处理策略

#### 结构化错误处理
```python
from enum import Enum
from typing import Union, Dict, Any

class ErrorType(Enum):
    VALIDATION_ERROR = "validation_error"
    NETWORK_ERROR = "network_error"
    PROCESSING_ERROR = "processing_error"

@node
def safe_api_call(url: str) -> Dict[str, Any]:
    """安全的API调用，包含完整错误处理"""
    try:
        response = make_http_request(url)
        return {
            'success': True,
            'data': response.json(),
            'status_code': response.status_code
        }
    except requests.ConnectionError:
        return {
            'success': False,
            'error_type': ErrorType.NETWORK_ERROR.value,
            'error_message': 'Network connection failed'
        }
    except requests.Timeout:
        return {
            'success': False,
            'error_type': ErrorType.NETWORK_ERROR.value,
            'error_message': 'Request timeout'
        }
    except Exception as e:
        return {
            'success': False,
            'error_type': ErrorType.PROCESSING_ERROR.value,
            'error_message': str(e)
        }
```

#### 错误传播和恢复
```python
@node
def handle_errors(api_result: Dict[str, Any]) -> Dict[str, Any]:
    """错误处理和恢复节点"""
    if not api_result.get('success', False):
        error_type = api_result.get('error_type')
        
        if error_type == ErrorType.NETWORK_ERROR.value:
            # 网络错误：使用缓存数据
            return {'data': get_cached_data(), 'source': 'cache'}
        elif error_type == ErrorType.VALIDATION_ERROR.value:
            # 验证错误：使用默认值
            return {'data': get_default_data(), 'source': 'default'}
        else:
            # 其他错误：向上传播
            return api_result
    
    return {'data': api_result['data'], 'source': 'api'}
```

## 并发编程最佳实践

### 1. 线程安全设计

#### 避免共享状态
```python
# ✅ 好的设计：无共享状态
@node
def process_item(item_data):
    result = expensive_computation(item_data)
    return {'processed': result}

# ❌ 危险设计：共享状态
global_counter = 0

@node
def unsafe_counter(item):
    global global_counter
    global_counter += 1  # 线程不安全
    return {'count': global_counter}
```

#### 使用线程本地存储
```python
import threading

# 线程本地存储
thread_local_data = threading.local()

@node
def thread_safe_counter(item):
    if not hasattr(thread_local_data, 'counter'):
        thread_local_data.counter = 0
    
    thread_local_data.counter += 1
    return {'thread_count': thread_local_data.counter}
```

### 2. 性能优化策略

#### 选择合适的执行模式
```python
# CPU密集型：使用进程池
cpu_intensive_flow = source.fan_out_to(
    [cpu_task1, cpu_task2], 
    executor='process',
    max_workers=4
)

# I/O密集型：使用线程池
io_intensive_flow = source.fan_out_to(
    [io_task1, io_task2], 
    executor='thread',
    max_workers=10
)
```

#### 批处理优化
```python
@node
def batch_processor(items: List[Any]) -> Dict[str, Any]:
    """批处理优化，减少系统调用开销"""
    batch_size = 100
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
    
    return {'results': results}
```

### 3. 资源管理

#### 连接池管理
```python
from contextlib import contextmanager

class ConnectionPool:
    def __init__(self, max_connections=10):
        self.pool = queue.Queue(max_connections)
        for _ in range(max_connections):
            self.pool.put(create_connection())
    
    @contextmanager
    def get_connection(self):
        conn = self.pool.get()
        try:
            yield conn
        finally:
            self.pool.put(conn)

@node
def database_operation(query: str) -> Dict[str, Any]:
    pool = get_connection_pool()
    
    with pool.get_connection() as conn:
        result = conn.execute(query)
        return {'result': result}
```

## 测试最佳实践

### 1. 单元测试
```python
import unittest
from unittest.mock import patch, MagicMock

class TestUserProcessing(unittest.TestCase):
    
    def test_validate_email_valid(self):
        """测试有效邮箱验证"""
        result = validate_email.run({'email': 'test@example.com'})
        self.assertTrue(result['is_valid'])
    
    def test_validate_email_invalid(self):
        """测试无效邮箱验证"""
        result = validate_email.run({'email': 'invalid-email'})
        self.assertFalse(result['is_valid'])
    
    @patch('requests.get')
    def test_api_call_with_mock(self, mock_get):
        """使用mock测试API调用"""
        mock_response = MagicMock()
        mock_response.json.return_value = {'data': 'test'}
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = api_call_node.run({'url': 'http://test.com'})
        self.assertTrue(result['success'])
```

### 2. 集成测试
```python
def test_complete_workflow():
    """测试完整工作流程"""
    workflow = (validate_input
                .then(process_data)
                .then(save_result))
    
    test_data = {'input': 'test_value'}
    result = workflow.run(test_data)
    
    assert 'saved' in result
    assert result['saved'] is True
```

### 3. 并发测试
```python
def test_concurrent_execution():
    """测试并发执行安全性"""
    import threading
    import time
    
    results = []
    
    def worker(worker_id):
        result = my_workflow.run({'id': worker_id})
        results.append(result)
    
    # 启动多个线程
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    # 等待完成
    for t in threads:
        t.join()
    
    # 验证结果
    assert len(results) == 10
    assert len(set(r['id'] for r in results)) == 10  # 所有结果都不同
```

## 监控和调试

### 1. 日志记录
```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@node
def logged_operation(data):
    """带日志记录的操作"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing data: {data}")
    
    try:
        result = complex_operation(data)
        logger.info(f"Operation successful: {result}")
        return {'result': result}
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        return {'error': str(e)}
```

### 2. 性能监控
```python
import time
from functools import wraps

def performance_monitor(func):
    """性能监控装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
        
        if isinstance(result, dict):
            result['_execution_time'] = execution_time
        
        return result
    return wrapper

@node
@performance_monitor
def monitored_operation(data):
    # 执行操作
    time.sleep(1)  # 模拟耗时操作
    return {'processed': data}
```

### 3. 调试工具
```python
@node
def debug_checkpoint(data, debug_enabled=False):
    """调试检查点"""
    if debug_enabled:
        import pdb; pdb.set_trace()  # 设置断点
    
    print(f"Debug: Current state = {data}")
    return data
```

## 部署和维护

### 1. 配置管理
```python
import os
from dataclasses import dataclass

@dataclass
class Config:
    database_url: str = os.getenv('DATABASE_URL', 'sqlite:///default.db')
    max_workers: int = int(os.getenv('MAX_WORKERS', '4'))
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'

# 全局配置
config = Config()

@node
def configured_operation(data):
    if config.debug:
        print(f"Debug: Processing {data}")
    
    return process_with_config(data, config)
```

### 2. 健康检查
```python
@node
def health_check():
    """系统健康检查"""
    checks = {
        'database': check_database_connection(),
        'cache': check_cache_connection(),
        'disk_space': check_disk_space(),
        'memory': check_memory_usage()
    }
    
    all_healthy = all(checks.values())
    
    return {
        'healthy': all_healthy,
        'checks': checks,
        'timestamp': time.time()
    }
```

### 3. 优雅关闭
```python
import signal
import sys

class GracefulShutdown:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)
        signal.signal(signal.SIGINT, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        print("Received shutdown signal, shutting down gracefully...")
        self.shutdown = True

# 全局关闭处理器
shutdown_handler = GracefulShutdown()

@node
def long_running_task(data):
    """支持优雅关闭的长运行任务"""
    results = []
    
    for item in data.get('items', []):
        if shutdown_handler.shutdown:
            break
        
        result = process_item(item)
        results.append(result)
    
    return {'results': results, 'completed': not shutdown_handler.shutdown}
```

## 总结

遵循这些最佳实践，您可以：

1. **构建可靠的系统**：通过错误处理和测试确保稳定性
2. **提高性能**：通过并发和优化策略提升效率  
3. **简化维护**：通过良好的架构和监控降低维护成本
4. **确保扩展性**：通过模块化设计支持系统增长

记住，最佳实践是指导原则，需要根据具体场景灵活应用。始终以系统的可读性、可维护性和性能为目标。