#!/usr/bin/env python3
"""
test_key_uniqueness_fix.py - 测试并行结果键重复处理修复的专项测试

主要测试场景：
1. 正常情况下无重复键的处理
2. 节点名称重复时的键唯一性处理
3. 异常处理路径的键重复处理
4. 混合场景（正常和异常结果的键冲突）
5. 边界情况和性能测试
"""

import time
import pytest
from typing import Dict, List, Any
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aetherflow import Node, ParallelResult, _generate_unique_result_key


# ============================================================================
# 测试辅助工具和模拟节点
# ============================================================================

@dataclass
class TestData:
    """测试数据结构"""
    value: int
    name: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class KeyTestHelper:
    """键重复测试的辅助工具类"""
    
    @staticmethod
    def create_simple_node(name: str, multiplier: int = 2, should_fail: bool = False) -> Node:
        """创建简单的测试节点"""
        def processor(data: TestData) -> TestData:
            if should_fail:
                raise ValueError(f"Node {name} intentionally failed")
            return TestData(
                value=data.value * multiplier,
                name=f"{name}_processed_{data.name}"
            )
        return Node(processor, name=name)
    
    @staticmethod
    def create_aggregator_node() -> Node:
        """创建聚合节点"""
        def aggregator(parallel_results: Dict[str, ParallelResult]) -> Dict[str, Any]:
            successful_count = sum(1 for r in parallel_results.values() if r.success)
            failed_count = len(parallel_results) - successful_count
            
            return {
                'total_results': len(parallel_results),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'result_keys': list(parallel_results.keys())
            }
        return Node(aggregator, name="key_aggregator")


# ============================================================================
# 测试 _generate_unique_result_key 函数
# ============================================================================

def test_generate_unique_key_no_conflict():
    """测试无冲突时的键生成"""
    print("\n=== 测试无冲突时的键生成 ===")
    
    existing_results = {
        'node1': 'some_result',
        'node2': 'another_result'
    }
    
    # 测试不冲突的键
    key = _generate_unique_result_key('node3', existing_results)
    assert key == 'node3', f"无冲突时应该返回原始键名，实际返回: {key}"
    
    print(f"✅ 无冲突键生成测试通过: {key}")


def test_generate_unique_key_with_conflict():
    """测试有冲突时的键生成"""
    print("\n=== 测试有冲突时的键生成 ===")
    
    existing_results = {
        'node1': 'result1',
        'node1[1]': 'result2',
        'node1[2]': 'result3'
    }
    
    # 测试冲突的键
    key = _generate_unique_result_key('node1', existing_results)
    assert key == 'node1[3]', f"冲突时应该返回node1[3]，实际返回: {key}"
    
    print(f"✅ 冲突键生成测试通过: {key}")


def test_generate_unique_key_empty_dict():
    """测试空字典时的键生成"""
    print("\n=== 测试空字典时的键生成 ===")
    
    key = _generate_unique_result_key('any_name', {})
    assert key == 'any_name', f"空字典时应该返回原始键名，实际返回: {key}"
    
    print(f"✅ 空字典键生成测试通过: {key}")


def test_generate_unique_key_sequential_conflicts():
    """测试连续冲突的键生成"""
    print("\n=== 测试连续冲突的键生成 ===")
    
    # 模拟连续添加相同名称的键
    existing_results = {}
    generated_keys = []
    
    for i in range(5):
        key = _generate_unique_result_key('duplicate', existing_results)
        generated_keys.append(key)
        existing_results[key] = f'result_{i}'
    
    expected_keys = ['duplicate', 'duplicate[1]', 'duplicate[2]', 'duplicate[3]', 'duplicate[4]']
    assert generated_keys == expected_keys, f"连续键生成不正确: {generated_keys} vs {expected_keys}"
    
    print(f"✅ 连续冲突键生成测试通过: {generated_keys}")


# ============================================================================
# 测试实际的并行执行键处理
# ============================================================================

def test_parallel_execution_unique_names():
    """测试并行执行中无重复名称的情况"""
    print("\n=== 测试并行执行无重复名称 ===")
    
    # 创建源节点
    def source_func(value: int) -> TestData:
        return TestData(value=value, name="source")
    
    source_node = Node(source_func, name="source")
    
    # 创建不同名称的目标节点
    target_nodes = [
        KeyTestHelper.create_simple_node(f"unique_node_{i}", multiplier=i+1)
        for i in range(3)
    ]
    
    # 执行并行处理
    fan_out_pipeline = source_node.fan_out_to(target_nodes)
    results = fan_out_pipeline(10)
    
    print(f"结果数量: {len(results)}")
    print(f"结果键: {list(results.keys())}")
    
    # 验证结果
    assert len(results) == 3, f"应该有3个结果，实际有{len(results)}个"
    
    expected_keys = {'unique_node_0', 'unique_node_1', 'unique_node_2'}
    actual_keys = set(results.keys())
    assert actual_keys == expected_keys, f"键应该无重复: {actual_keys} vs {expected_keys}"
    
    # 验证所有结果都成功
    for key, result in results.items():
        assert result.success, f"节点{key}执行应该成功"
        print(f"  {key}: value={result.result.value}, success={result.success}")
    
    print("✅ 并行执行无重复名称测试通过")


def test_parallel_execution_duplicate_names():
    """测试并行执行中重复名称的键处理"""
    print("\n=== 测试并行执行重复名称键处理 ===")
    
    # 创建源节点
    def source_func(value: int) -> TestData:
        return TestData(value=value, name="source")
    
    source_node = Node(source_func, name="source")
    
    # 创建相同名称的目标节点
    target_nodes = [
        KeyTestHelper.create_simple_node("duplicate_node", multiplier=i+1)
        for i in range(4)
    ]
    
    # 执行并行处理
    fan_out_pipeline = source_node.fan_out_to(target_nodes)
    results = fan_out_pipeline(5)
    
    print(f"结果数量: {len(results)}")
    print(f"结果键: {sorted(results.keys())}")
    
    # 验证结果
    assert len(results) == 4, f"应该有4个结果，实际有{len(results)}个"
    
    # 验证键的唯一性和预期格式
    expected_keys = {'duplicate_node', 'duplicate_node[1]', 'duplicate_node[2]', 'duplicate_node[3]'}
    actual_keys = set(results.keys())
    assert actual_keys == expected_keys, f"重复名称键处理不正确: {actual_keys} vs {expected_keys}"
    
    # 验证所有结果都成功且值不同（不同的multiplier）
    values = []
    for key, result in results.items():
        assert result.success, f"节点{key}执行应该成功"
        values.append(result.result.value)
        print(f"  {key}: value={result.result.value}, success={result.success}")
    
    # 验证值的唯一性（每个节点有不同的multiplier）
    expected_values = {5, 10, 15, 20}  # 5*(1,2,3,4)
    actual_values = set(values)
    assert actual_values == expected_values, f"结果值应该不同: {actual_values} vs {expected_values}"
    
    print("✅ 并行执行重复名称键处理测试通过")


def test_parallel_execution_with_failures():
    """测试并行执行中包含异常的键处理"""
    print("\n=== 测试并行执行异常键处理 ===")
    
    # 创建源节点
    def source_func(value: int) -> TestData:
        return TestData(value=value, name="source")
    
    source_node = Node(source_func, name="source")
    
    # 创建混合节点：正常和失败节点都使用相同名称
    target_nodes = [
        KeyTestHelper.create_simple_node("mixed_node", multiplier=2, should_fail=False),
        KeyTestHelper.create_simple_node("mixed_node", multiplier=3, should_fail=True),
        KeyTestHelper.create_simple_node("mixed_node", multiplier=4, should_fail=False),
        KeyTestHelper.create_simple_node("mixed_node", multiplier=5, should_fail=True),
    ]
    
    # 执行并行处理
    fan_out_pipeline = source_node.fan_out_to(target_nodes)
    results = fan_out_pipeline(6)
    
    print(f"结果数量: {len(results)}")
    print(f"结果键: {sorted(results.keys())}")
    
    # 验证结果
    assert len(results) == 4, f"应该有4个结果，实际有{len(results)}个"
    
    # 验证键的唯一性
    expected_keys = {'mixed_node', 'mixed_node[1]', 'mixed_node[2]', 'mixed_node[3]'}
    actual_keys = set(results.keys())
    assert actual_keys == expected_keys, f"混合场景键处理不正确: {actual_keys} vs {expected_keys}"
    
    # 统计成功和失败结果
    successful_results = []
    failed_results = []
    
    for key, result in results.items():
        if result.success:
            successful_results.append((key, result))
            print(f"  ✅ {key}: value={result.result.value}")
        else:
            failed_results.append((key, result))
            print(f"  ❌ {key}: error={result.error}")
    
    # 验证结果统计
    assert len(successful_results) == 2, f"应该有2个成功结果，实际有{len(successful_results)}个"
    assert len(failed_results) == 2, f"应该有2个失败结果，实际有{len(failed_results)}个"
    
    # 验证成功结果的值
    success_values = {result.result.value for _, result in successful_results}
    expected_success_values = {12, 24}  # 6*2, 6*4
    assert success_values == expected_success_values, f"成功结果值不匹配: {success_values} vs {expected_success_values}"
    
    # 验证失败结果包含错误信息
    for key, result in failed_results:
        assert result.error is not None, f"失败结果{key}应该包含错误信息"
        assert "intentionally failed" in result.error, f"失败结果{key}应该包含预期的错误信息"
    
    print("✅ 并行执行异常键处理测试通过")


def test_fan_out_fan_in_key_handling():
    """测试完整的fan_out_fan_in流程中的键处理"""
    print("\n=== 测试fan_out_fan_in键处理 ===")
    
    # 创建源节点
    def source_func(value: int) -> TestData:
        return TestData(value=value, name="source")
    
    source_node = Node(source_func, name="source")
    
    # 创建重复名称的目标节点
    target_nodes = [
        KeyTestHelper.create_simple_node("fanout_target", multiplier=i+1)
        for i in range(3)
    ]
    
    # 创建聚合节点
    aggregator_node = KeyTestHelper.create_aggregator_node()
    
    # 执行完整的fan_out_fan_in流程
    pipeline = source_node.fan_out_in(target_nodes, aggregator_node)
    result = pipeline(8)
    
    print(f"聚合结果: {result}")
    
    # 验证聚合结果
    assert isinstance(result, dict), "聚合结果应该是字典"
    assert result['total_results'] == 3, f"应该处理3个结果，实际处理了{result['total_results']}个"
    assert result['successful_count'] == 3, f"应该有3个成功结果，实际有{result['successful_count']}个"
    assert result['failed_count'] == 0, f"应该有0个失败结果，实际有{result['failed_count']}个"
    
    # 验证结果键的格式
    result_keys = set(result['result_keys'])
    expected_keys = {'fanout_target', 'fanout_target[1]', 'fanout_target[2]'}
    assert result_keys == expected_keys, f"聚合器接收到的键不正确: {result_keys} vs {expected_keys}"
    
    print(f"✅ fan_out_fan_in键处理测试通过，处理的键: {result['result_keys']}")


# ============================================================================
# 性能和边界情况测试
# ============================================================================

def test_large_scale_duplicate_keys():
    """测试大量重复键的性能"""
    print("\n=== 测试大量重复键性能 ===")
    
    start_time = time.time()
    
    # 模拟大量重复键的场景
    existing_results = {}
    generated_keys = []
    
    # 生成100个重复键
    for i in range(100):
        key = _generate_unique_result_key('performance_test', existing_results)
        generated_keys.append(key)
        existing_results[key] = f'result_{i}'
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"生成100个键耗时: {execution_time:.4f}秒")
    print(f"前10个键: {generated_keys[:10]}")
    print(f"后10个键: {generated_keys[-10:]}")
    
    # 验证性能（应该在合理时间内完成）
    assert execution_time < 1.0, f"性能测试超时: {execution_time}秒"
    
    # 验证键的唯一性
    assert len(set(generated_keys)) == 100, "所有生成的键应该都是唯一的"
    
    # 验证键的格式
    assert generated_keys[0] == 'performance_test', "第一个键应该是原始名称"
    assert generated_keys[1] == 'performance_test[1]', "第二个键应该是带[1]后缀"
    assert generated_keys[99] == 'performance_test[99]', "第100个键应该是带[99]后缀"
    
    print("✅ 大量重复键性能测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    
    # 测试空字符串键名
    key1 = _generate_unique_result_key('', {})
    assert key1 == '', "空字符串键名应该正常处理"
    
    # 测试包含特殊字符的键名
    special_name = 'node@#$%^&*()'
    key2 = _generate_unique_result_key(special_name, {})
    assert key2 == special_name, "特殊字符键名应该正常处理"
    
    # 测试极长的键名
    long_name = 'very_long_node_name_' * 10
    key3 = _generate_unique_result_key(long_name, {})
    assert key3 == long_name, "极长键名应该正常处理"
    
    # 测试包含数字的键名冲突
    existing = {'node123': 'result'}
    key4 = _generate_unique_result_key('node123', existing)
    assert key4 == 'node123[1]', "包含数字的键名冲突应该正常处理"
    
    print("✅ 边界情况测试通过")


if __name__ == "__main__":
    print("=== 并行结果键重复处理修复测试 ===")
    
    try:
        # 测试核心函数
        test_generate_unique_key_no_conflict()
        test_generate_unique_key_with_conflict()
        test_generate_unique_key_empty_dict()
        test_generate_unique_key_sequential_conflicts()
        
        # 测试实际并行执行
        test_parallel_execution_unique_names()
        test_parallel_execution_duplicate_names()
        test_parallel_execution_with_failures()
        test_fan_out_fan_in_key_handling()
        
        # 测试性能和边界情况
        test_large_scale_duplicate_keys()
        test_edge_cases()
        
        print("\n🎉 所有键重复处理修复测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()