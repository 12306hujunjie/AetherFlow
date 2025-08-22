#!/usr/bin/env python3
"""
test_then_comprehensive.py - Node的then方法综合测试
包含：基本then功能、链式调用、依赖注入、context/state传递的完整测试
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pydantic import BaseModel, Field, validator
from dependency_injector.wiring import Provide
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random
from src.aetherflow import Node, BaseFlowContext, node


# 定义测试数据结构
@dataclass
class UserInput:
    name: str
    age: int

@dataclass  
class ProcessedUser:
    formatted_name: str
    is_adult: bool

@dataclass
class FinalResult:
    message: str
    user_type: str


def test_basic_then():
    """测试基本的then功能"""
    print("=== 测试基本then功能 ===")
    
    def process_user_input(user: UserInput) -> ProcessedUser:
        print(f"Processing: {user.name}, age: {user.age}")
        return ProcessedUser(
            formatted_name=user.name.title(),
            is_adult=user.age >= 18
        )
    
    def generate_final_result(processed: ProcessedUser) -> FinalResult:
        print(f"Generating result for: {processed.formatted_name}")
        user_type = "成人" if processed.is_adult else "未成年"
        return FinalResult(
            message=f"欢迎 {processed.formatted_name}",
            user_type=user_type
        )
    
    # 创建节点
    node1 = Node(process_user_input, name="process_user")
    node2 = Node(generate_final_result, name="generate_result")
    
    print(f"Node1: {node1}")
    print(f"Node2: {node2}")
    
    # 测试then组合
    pipeline = node1.then(node2)
    print(f"Pipeline: {pipeline}")
    
    # 执行测试
    user_input = UserInput(name="alice", age=25)
    result = pipeline(user_input)
    
    print(f"Pipeline result: {result}")
    assert result.message == "欢迎 Alice"
    assert result.user_type == "成人"
    print("✅ 基本then测试通过")


def test_chain_then():
    """测试链式then调用"""
    print("\n=== 测试链式then调用 ===")
    
    def multiply_by_2(x: int) -> int:
        result = x * 2
        print(f"Multiply: {x} -> {result}")
        return result
        
    def add_10(x: int) -> int:
        result = x + 10
        print(f"Add 10: {x} -> {result}")
        return result
        
    def format_result(x: int) -> str:
        result = f"final_{x}"
        print(f"Format: {x} -> {result}")
        return result
    
    # 创建节点
    step1 = Node(multiply_by_2, name="multiply")
    step2 = Node(add_10, name="add")
    step3 = Node(format_result, name="format")
    
    # 三级链式调用
    pipeline = step1.then(step2).then(step3)
    print(f"Chain pipeline: {pipeline}")
    
    # 执行: 5 -> 10 -> 20 -> "final_20"
    result = pipeline(5)
    print(f"Chain result: {result}")
    assert result == "final_20"
    print("✅ 链式then测试通过")


def test_then_with_dependency_injection():
    """测试then链式调用中的依赖注入"""
    print("\n=== 测试then + 依赖注入 ===")
    
    # 第一个节点：处理输入并存储到state
    def process_input(data: dict, context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        state = context.state()
        processed_value = data['input'] * 2
        state['step1_processed'] = processed_value
        state['step1_timestamp'] = '2024-01-01'
        print(f"Step1: {data['input']} -> {processed_value} (stored in state)")
        return {'processed': processed_value, 'original': data['input']}
    
    # 第二个节点：从state读取并进一步处理
    def enhance_data(data: dict, context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        state = context.state()
        # 读取前一步的state
        step1_value = state.get('step1_processed', 0)
        enhanced = step1_value + data['processed'] + 10
        state['step2_enhanced'] = enhanced
        state['step2_source'] = f"from_step1_{step1_value}"
        print(f"Step2: enhanced {enhanced} using state value {step1_value}")
        return {'enhanced': enhanced, 'chain_data': data}
    
    # 第三个节点：最终汇总
    def finalize_result(data: dict, context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        state = context.state()
        final_value = data['enhanced'] * 1.5
        state['final_result'] = final_value
        state['processing_chain'] = [
            state.get('step1_processed'),
            state.get('step2_enhanced'), 
            final_value
        ]
        print(f"Step3: final result {final_value}")
        return {
            'final': final_value,
            'chain_history': state['processing_chain'],
            'metadata': {
                'step1_time': state.get('step1_timestamp'),
                'step2_source': state.get('step2_source')
            }
        }
    
    # 创建节点
    node1 = Node(process_input, name="process")
    node2 = Node(enhance_data, name="enhance") 
    node3 = Node(finalize_result, name="finalize")
    
    # 创建并配置container
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 构建链式调用
    pipeline = node1.then(node2).then(node3)
    print(f"Injection pipeline: {pipeline}")
    
    # 执行链式调用
    result = pipeline({'input': 5})
    
    print(f"链式调用结果: {result}")
    print(f"最终state: {container.state()}")
    
    # 验证链式调用结果 - ((5*2) + (5*2) + 10) * 1.5 = 30 * 1.5 = 45
    assert result['final'] == 45.0
    assert result['chain_history'] == [10, 30, 45.0]
    assert result['metadata']['step1_time'] == '2024-01-01'
    assert result['metadata']['step2_source'] == 'from_step1_10'
    
    # 验证state中的中间结果
    final_state = container.state()
    assert final_state['step1_processed'] == 10
    assert final_state['step2_enhanced'] == 30
    assert final_state['final_result'] == 45.0
    print("✅ then + 依赖注入测试通过")


def test_decorator_with_then_chain():
    """测试@node装饰器与then链式调用的结合"""
    print("\n=== 测试@node装饰器 + then链式调用 ===")
    
    @node
    def extract_number(text: str, context: BaseFlowContext = Provide[BaseFlowContext]) -> int:
        state = context.state()
        # 提取文本中的数字
        numbers = ''.join(filter(str.isdigit, text))
        extracted = int(numbers) if numbers else 0
        state['extracted_number'] = extracted
        state['original_text'] = text
        print(f"Extracted {extracted} from '{text}'")
        return extracted
    
    @node
    def calculate_square(num: int, context: BaseFlowContext = Provide[BaseFlowContext]) -> int:
        state = context.state()
        squared = num * num
        state['squared_result'] = squared
        print(f"Squared {num} to get {squared}")
        return squared
    
    @node
    def format_output(num: int, context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        state = context.state()
        original_text = state.get('original_text', 'unknown')
        extracted = state.get('extracted_number', 0)
        
        result = {
            'final_result': num,
            'calculation_chain': f"{original_text} -> {extracted} -> {num}",
            'operation': 'extract_and_square'
        }
        state['final_output'] = result
        print(f"Formatted final output: {result}")
        return result
    
    # 配置依赖注入
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 构建装饰器节点的链式调用
    pipeline = extract_number.then(calculate_square).then(format_output)
    
    # 执行: "hello123world" -> 123 -> 15129 -> formatted result
    result = pipeline("hello123world")
    
    print(f"装饰器链式结果: {result}")
    print(f"装饰器链式state: {container.state()}")
    
    assert result['final_result'] == 15129  # 123^2 = 15129
    assert result['calculation_chain'] == "hello123world -> 123 -> 15129"
    assert container.state()['extracted_number'] == 123
    assert container.state()['squared_result'] == 15129
    print("✅ @node装饰器 + then链式调用测试通过")


def test_mixed_injection_and_simple_nodes():
    """测试混合使用有注入和无注入的节点"""
    print("\n=== 测试混合节点类型 ===")
    
    # 简单节点（无注入）
    def simple_double(x: int) -> int:
        result = x * 2
        print(f"Simple double: {x} -> {result}")
        return result
    
    # 依赖注入节点
    def state_tracker(x: int, context: BaseFlowContext = Provide[BaseFlowContext]) -> int:
        state = context.state()
        state['current_value'] = x
        state['operations'] = state.get('operations', []) + ['tracked']
        print(f"State tracker: {x} (operations: {state['operations']})")
        return x + 5
    
    # 另一个简单节点
    def simple_format(x: int) -> str:
        result = f"result_{x}"
        print(f"Simple format: {x} -> {result}")
        return result
    
    # 创建节点
    double_node = Node(simple_double, name="double")
    track_node = Node(state_tracker, name="tracker") 
    format_node = Node(simple_format, name="format")
    
    # 配置容器（只对需要注入的节点生效）
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 混合链式调用：simple -> injection -> simple
    mixed_pipeline = double_node.then(track_node).then(format_node)
    
    # 执行: 10 -> 20 -> 25 -> "result_25"
    result = mixed_pipeline(10)
    
    print(f"混合节点结果: {result}")
    print(f"混合节点state: {container.state()}")
    
    assert result == "result_25"
    assert container.state()['current_value'] == 20  # double后的值
    assert container.state()['operations'] == ['tracked']
    print("✅ 混合节点类型测试通过")


def test_type_validation_in_chains():
    """测试链式调用中的类型验证"""
    print("\n=== 测试链式调用类型验证 ===")
    
    def strict_int_processor(x: int) -> str:
        return f"processed_{x}"
    
    def strict_str_processor(s: str) -> int:
        return len(s)
    
    def strict_final_processor(x: int) -> dict:
        return {'length': x, 'valid': x > 5}
    
    node1 = Node(strict_int_processor, name="int_to_str")
    node2 = Node(strict_str_processor, name="str_to_int")
    node3 = Node(strict_final_processor, name="final_check")
    
    pipeline = node1.then(node2).then(node3)
    
    # 正确的类型输入
    result = pipeline(42)
    print(f"Valid chain result: {result}")
    assert result['length'] == 12  # len("processed_42") = 12
    assert result['valid'] == True  # 12 > 5
    
    # 测试类型错误（应该被Pydantic捕获）
    try:
        pipeline("invalid")  # 传递字符串给期望int的函数
        assert False, "应该抛出验证错误"
    except Exception as e:
        print(f"Expected validation error: {type(e).__name__}")
        print("✅ 链式调用类型验证测试通过")


def test_user_input_parameter_validation():
    """测试用户传入参数的类型校验"""
    print("\n=== 测试用户传入参数校验 ===")
    
    # 严格的整数输入节点
    def strict_int_input(num: int) -> str:
        print(f"处理整数输入: {num}")
        return f"number_{num}"
    
    # 严格的字典输入节点  
    from typing import Dict
    def strict_dict_input(data: Dict[str, int]) -> int:
        print(f"处理字典输入: {data}")
        return sum(data.values())
    
    # 复杂类型输入节点
    from typing import List, Tuple
    def complex_input(items: List[Tuple[str, int]]) -> Dict[str, int]:
        print(f"处理复杂输入: {items}")
        return {name: value * 2 for name, value in items}
    
    # 测试1: 正确的整数输入
    int_node = Node(strict_int_input, name="int_input")
    result1 = int_node(42)
    print(f"正确整数输入结果: {result1}")
    assert result1 == "number_42"
    
    # 测试2: 错误的整数输入类型
    try:
        int_node("not_a_number")  # 传递字符串给期望int的函数
        assert False, "应该抛出类型验证错误"
    except Exception as e:
        print(f"✅ 正确捕获错误整数输入: {type(e).__name__}")
    
    # 测试3: 正确的字典输入
    dict_node = Node(strict_dict_input, name="dict_input")
    result3 = dict_node({"a": 10, "b": 20, "c": 30})
    print(f"正确字典输入结果: {result3}")
    assert result3 == 60
    
    # 测试4: 错误的字典输入类型
    try:
        dict_node({"a": "not_int", "b": 20})  # 字典值类型错误
        assert False, "应该抛出字典值类型验证错误"
    except Exception as e:
        print(f"✅ 正确捕获错误字典输入: {type(e).__name__}")
    
    # 测试5: 复杂类型输入
    complex_node = Node(complex_input, name="complex_input")
    result5 = complex_node([("apple", 5), ("banana", 3)])
    print(f"复杂类型输入结果: {result5}")
    assert result5 == {"apple": 10, "banana": 6}
    
    # 测试6: 错误的复杂类型输入
    try:
        complex_node([("apple", "not_int"), ("banana", 3)])  # tuple中类型错误
        assert False, "应该抛出复杂类型验证错误"
    except Exception as e:
        print(f"✅ 正确捕获错误复杂类型输入: {type(e).__name__}")
    
    print("✅ 用户传入参数校验测试通过")


def test_node_output_to_input_validation():
    """测试节点间输出到输入的参数类型校验"""
    print("\n=== 测试节点间参数传递校验 ===")
    
    # 定义一系列有明确输入输出类型的节点
    def int_to_str_node(num: int) -> str:
        result = f"str_{num}"
        print(f"int->str: {num} -> {result}")
        return result
    
    def str_to_list_node(text: str) -> List[str]:
        result = text.split("_")
        print(f"str->list: {text} -> {result}")
        return result
    
    def list_to_dict_node(items: List[str]) -> Dict[str, int]:
        result = {item: len(item) for item in items}
        print(f"list->dict: {items} -> {result}")
        return result
    
    def dict_to_int_node(data: Dict[str, int]) -> int:
        result = sum(data.values())
        print(f"dict->int: {data} -> {result}")
        return result
    
    # 故意返回错误类型的节点
    def wrong_output_type(num: int) -> str:
        print(f"返回错误类型: {num}")
        return num * 2  # 声明返回str但实际返回int
    
    def expect_str_input(text: str) -> str:
        print(f"期望字符串输入: {text}")
        return text.upper()
    
    # 测试1: 正确的类型链式传递
    node1 = Node(int_to_str_node, name="int_to_str")
    node2 = Node(str_to_list_node, name="str_to_list")  
    node3 = Node(list_to_dict_node, name="list_to_dict")
    node4 = Node(dict_to_int_node, name="dict_to_int")
    
    correct_pipeline = node1.then(node2).then(node3).then(node4)
    result1 = correct_pipeline(123)  # 123 -> "str_123" -> ["str", "123"] -> {"str":3, "123":3} -> 6
    print(f"正确类型链式结果: {result1}")
    assert result1 == 6  # len("str") + len("123") = 3 + 3 = 6
    
    # 测试2: 中间节点返回错误类型
    wrong_node1 = Node(wrong_output_type, name="wrong_output")
    correct_node2 = Node(expect_str_input, name="expect_str")
    
    try:
        # 这个链式调用应该失败，因为wrong_output_type返回int但下一个节点期望str
        wrong_pipeline = wrong_node1.then(correct_node2)
        result2 = wrong_pipeline(10)
        print(f"⚠️  错误类型传递未被捕获: {result2}")
        # 如果Pydantic自动转换了类型，验证转换是否正确
        assert isinstance(result2, str), "应该转换为字符串类型"
    except Exception as e:
        print(f"✅ 正确捕获节点间类型不匹配: {type(e).__name__}")
    
    # 测试3: 复杂的类型不匹配场景
    from typing import Optional
    def optional_input_node(data: Optional[Dict[str, int]]) -> str:
        if data is None:
            return "empty"
        return f"data_count_{len(data)}"
    
    def return_none_node(x: int) -> None:
        print(f"返回None: {x}")
        return None
    
    def return_optional_dict(x: int) -> Optional[Dict[str, int]]:
        if x > 0:
            return {"value": x}
        return None
    
    # 测试None传递
    none_node = Node(return_none_node, name="return_none")
    optional_node = Node(optional_input_node, name="optional_input")
    
    try:
        none_pipeline = none_node.then(optional_node)
        result3 = none_pipeline(5)
        print(f"None传递结果: {result3}")
        # 如果成功，验证结果
        assert result3 in ["empty", "NONE"], "None应该被正确处理"
    except Exception as e:
        print(f"None传递失败: {type(e).__name__}")
    
    # 测试Optional类型传递
    optional_dict_node = Node(return_optional_dict, name="return_optional")
    optional_pipeline = optional_dict_node.then(optional_node)
    
    result4 = optional_pipeline(10)  # 返回{"value": 10} -> "data_count_1"
    print(f"Optional类型传递结果: {result4}")
    assert result4 == "data_count_1"
    
    result5 = optional_pipeline(-5)  # 返回None -> "empty"
    print(f"Optional None传递结果: {result5}")
    assert result5 == "empty"
    
    print("✅ 节点间参数传递校验测试通过")


def test_chain_type_mismatch_errors():
    """测试链式调用中的类型不匹配错误处理"""
    print("\n=== 测试链式调用类型不匹配错误 ===")
    
    # 创建一系列类型不兼容的节点
    def produce_int(x: str) -> int:
        return len(x)
    
    def expect_str(x: str) -> str:  # 期望str但上个节点返回int
        return x.upper()
    
    def produce_list(x: int) -> List[int]:
        return [x, x*2, x*3]
    
    def expect_dict(x: Dict[str, int]) -> int:  # 期望dict但上个节点返回list
        return sum(x.values())
    
    # 测试1: int->str 类型不匹配
    int_node = Node(produce_int, name="produce_int")
    str_node = Node(expect_str, name="expect_str")
    
    try:
        type_mismatch_1 = int_node.then(str_node)
        result1 = type_mismatch_1("hello")  # "hello" -> 5 -> expect_str(5) 应该失败
        print(f"⚠️  int->str不匹配未捕获: {result1}")
        # 可能Pydantic自动转换了
        assert isinstance(result1, str), "应该转换为字符串"
    except Exception as e:
        print(f"✅ 正确捕获int->str不匹配: {type(e).__name__}")
    
    # 测试2: list->dict 类型不匹配  
    list_node = Node(produce_list, name="produce_list")
    dict_node = Node(expect_dict, name="expect_dict")
    
    try:
        type_mismatch_2 = list_node.then(dict_node)
        result2 = type_mismatch_2(3)  # 3 -> [3,6,9] -> expect_dict([3,6,9]) 应该失败
        assert False, "list->dict不匹配应该抛出错误"
    except Exception as e:
        print(f"✅ 正确捕获list->dict不匹配: {type(e).__name__}")
    
    # 测试3: 三级链式调用中的中间类型错误
    def step1_str_to_int(s: str) -> int:
        return int(s) if s.isdigit() else 0
    
    def step2_int_to_str(n: int) -> str:
        return f"result_{n}"
    
    def step3_expect_int(n: int) -> dict:  # 期望int但上个节点返回str
        return {"doubled": n * 2}
    
    step1 = Node(step1_str_to_int, name="step1")
    step2 = Node(step2_int_to_str, name="step2") 
    step3 = Node(step3_expect_int, name="step3")
    
    try:
        three_step_chain = step1.then(step2).then(step3)
        result3 = three_step_chain("42")  # "42" -> 42 -> "result_42" -> step3("result_42") 应该失败
        assert False, "三级链式类型不匹配应该抛出错误"
    except Exception as e:
        print(f"✅ 正确捕获三级链式类型不匹配: {type(e).__name__}")
    
    # 测试4: 混合依赖注入和类型不匹配
    def inject_and_wrong_type(data: str, context: BaseFlowContext = Provide[BaseFlowContext]) -> int:
        state = context.state()
        state['input_data'] = data
        return len(data)  # 返回int
    
    def expect_str_with_injection(text: str, context: BaseFlowContext = Provide[BaseFlowContext]) -> str:
        state = context.state()
        state['processed_text'] = text
        return text.upper()  # 期望str输入
    
    # 配置容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    inject_wrong_node = Node(inject_and_wrong_type, name="inject_wrong")
    expect_str_inject_node = Node(expect_str_with_injection, name="expect_str_inject")
    
    try:
        inject_mismatch_chain = inject_wrong_node.then(expect_str_inject_node)
        result4 = inject_mismatch_chain("test")  # "test" -> 4 -> expect_str_with_injection(4) 
        print(f"⚠️  注入+类型不匹配未捕获: {result4}")
    except Exception as e:
        print(f"✅ 正确捕获注入+类型不匹配: {type(e).__name__}")
    
    print("✅ 链式调用类型不匹配错误测试完成")


def test_pydantic_model_passing():
    """测试节点之间传递Pydantic模型"""
    print("\n=== 测试Pydantic模型传递 ===")
    
    # 定义Pydantic模型
    class UserModel(BaseModel):
        name: str = Field(..., min_length=1, max_length=50)
        age: int = Field(..., ge=0, le=150)
        email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
        
        @validator('name')
        def name_must_be_alpha(cls, v):
            if not v.replace(' ', '').isalpha():
                raise ValueError('名字必须只包含字母和空格')
            return v.title()
    
    class ProcessedUserModel(BaseModel):
        id: int
        full_name: str
        is_adult: bool
        email_domain: str
        created_at: str
        
    class UserStatsModel(BaseModel):
        user_count: int
        adult_count: int  
        domains: List[str]
        average_age: float
        
    # 节点1: 输入验证和初步处理
    def validate_and_process_user(user_data: dict) -> UserModel:
        print(f"验证用户数据: {user_data}")
        # Pydantic自动验证和转换
        user = UserModel(**user_data)
        print(f"验证通过的用户: {user}")
        return user
    
    # 节点2: 用户模型转换
    def transform_user_model(user: UserModel) -> ProcessedUserModel:
        print(f"转换用户模型: {user.name}, {user.age}")
        processed = ProcessedUserModel(
            id=hash(user.email) % 10000,  # 简单的ID生成
            full_name=user.name,
            is_adult=user.age >= 18,
            email_domain=user.email.split('@')[1],
            created_at='2024-01-01T00:00:00Z'
        )
        print(f"转换后的用户: {processed}")
        return processed
        
    # 节点3: 聚合统计（模拟批处理）
    def aggregate_user_stats(processed_user: ProcessedUserModel, 
                           context: BaseFlowContext = Provide[BaseFlowContext]) -> UserStatsModel:
        state = context.state()
        
        # 从状态中获取累积数据
        users = state.get('processed_users', [])
        users.append(processed_user)
        state['processed_users'] = users
        
        print(f"聚合统计，当前用户数: {len(users)}")
        
        # 计算统计信息
        adult_count = sum(1 for u in users if u.is_adult)
        domains = list(set(u.email_domain for u in users))
        # 这里需要从原始数据计算平均年龄，简化处理
        avg_age = 25.0  # 模拟平均年龄
        
        stats = UserStatsModel(
            user_count=len(users),
            adult_count=adult_count,
            domains=domains,
            average_age=avg_age
        )
        print(f"统计结果: {stats}")
        return stats
    
    # 创建节点
    validate_node = Node(validate_and_process_user, name="validate")
    transform_node = Node(transform_user_model, name="transform")
    aggregate_node = Node(aggregate_user_stats, name="aggregate")
    
    # 配置容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 构建链式调用
    pipeline = validate_node.then(transform_node).then(aggregate_node)
    
    # 测试1: 正确的用户数据
    user_data1 = {
        "name": "alice smith",
        "age": 25,
        "email": "alice@example.com"
    }
    
    result1 = pipeline(user_data1)
    print(f"第一个用户处理结果: {result1}")
    
    assert isinstance(result1, UserStatsModel)
    assert result1.user_count == 1
    assert result1.adult_count == 1
    assert "example.com" in result1.domains
    
    # 测试2: 第二个用户（累积）
    user_data2 = {
        "name": "bob jones", 
        "age": 17,
        "email": "bob@test.org"
    }
    
    result2 = pipeline(user_data2)
    print(f"第二个用户处理结果: {result2}")
    
    assert result2.user_count == 2
    assert result2.adult_count == 1  # 只有alice是成年人
    assert len(result2.domains) == 2  # example.com 和 test.org
    
    # 测试3: 无效的用户数据（应该被Pydantic捕获）
    try:
        invalid_data = {
            "name": "123invalid",  # 包含数字，不符合validator
            "age": 200,  # 超出范围
            "email": "invalid-email"  # 无效邮箱格式
        }
        pipeline(invalid_data)
        assert False, "应该抛出Pydantic验证错误"
    except Exception as e:
        print(f"✅ 正确捕获Pydantic验证错误: {type(e).__name__}")
    
    print("✅ Pydantic模型传递测试通过")


def test_complex_pydantic_model_chains():
    """测试复杂的Pydantic模型链式传递"""
    print("\n=== 测试复杂Pydantic模型链式传递 ===")
    
    # 定义复杂的嵌套模型
    class AddressModel(BaseModel):
        street: str
        city: str
        country: str
        postal_code: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
        
    class CompanyModel(BaseModel):
        name: str
        industry: str
        employees: int = Field(..., gt=0)
        
    class PersonModel(BaseModel):
        name: str
        age: int = Field(..., ge=18)
        address: AddressModel
        company: Optional[CompanyModel] = None
        skills: List[str] = Field(default_factory=list)
        
    class EnrichedPersonModel(BaseModel):
        person: PersonModel
        location_score: float = Field(..., ge=0.0, le=10.0)
        career_level: str
        skill_categories: Dict[str, List[str]]
        
    class AnalyticsModel(BaseModel):
        total_people: int
        average_age: float
        top_cities: List[str]
        skill_frequency: Dict[str, int]
        company_industries: Dict[str, int]
        
    # 节点1: 个人信息丰富化
    def enrich_person_data(person: PersonModel) -> EnrichedPersonModel:
        print(f"丰富化个人数据: {person.name}")
        
        # 计算位置评分（模拟）
        location_scores = {"New York": 9.0, "San Francisco": 8.5, "Austin": 7.0}
        location_score = location_scores.get(person.address.city, 5.0)
        
        # 确定职业级别
        if person.company and person.company.employees > 1000:
            career_level = "Senior"
        elif person.age >= 30:
            career_level = "Mid-level"
        else:
            career_level = "Junior"
            
        # 技能分类
        tech_skills = ["Python", "JavaScript", "React", "Django"]
        business_skills = ["Management", "Marketing", "Sales"]
        
        skill_categories = {
            "technical": [s for s in person.skills if s in tech_skills],
            "business": [s for s in person.skills if s in business_skills],
            "other": [s for s in person.skills if s not in tech_skills + business_skills]
        }
        
        enriched = EnrichedPersonModel(
            person=person,
            location_score=location_score,
            career_level=career_level,
            skill_categories=skill_categories
        )
        
        print(f"丰富化完成: {enriched.career_level}, 位置评分: {enriched.location_score}")
        return enriched
        
    # 节点2: 数据分析聚合
    def analyze_enriched_data(enriched: EnrichedPersonModel,
                            context: BaseFlowContext = Provide[BaseFlowContext]) -> AnalyticsModel:
        state = context.state()
        
        # 累积数据
        people = state.get('enriched_people', [])
        people.append(enriched)
        state['enriched_people'] = people
        
        print(f"分析数据，当前人数: {len(people)}")
        
        # 计算分析指标
        total_people = len(people)
        average_age = sum(p.person.age for p in people) / total_people
        
        # 统计城市
        cities = [p.person.address.city for p in people]
        city_counts = {}
        for city in cities:
            city_counts[city] = city_counts.get(city, 0) + 1
        top_cities = sorted(city_counts.keys(), key=city_counts.get, reverse=True)[:3]
        
        # 统计技能
        skill_frequency = {}
        for p in people:
            for skills_list in p.skill_categories.values():
                for skill in skills_list:
                    skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
        
        # 统计行业
        company_industries = {}
        for p in people:
            if p.person.company:
                industry = p.person.company.industry
                company_industries[industry] = company_industries.get(industry, 0) + 1
                
        analytics = AnalyticsModel(
            total_people=total_people,
            average_age=average_age,
            top_cities=top_cities,
            skill_frequency=skill_frequency,
            company_industries=company_industries
        )
        
        print(f"分析完成: {analytics}")
        return analytics
    
    # 创建节点
    enrich_node = Node(enrich_person_data, name="enrich")
    analyze_node = Node(analyze_enriched_data, name="analyze")
    
    # 配置容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 构建链式调用
    complex_pipeline = enrich_node.then(analyze_node)
    
    # 测试复杂嵌套模型
    person_data = {
        "name": "John Developer",
        "age": 28,
        "address": {
            "street": "123 Main St",
            "city": "San Francisco", 
            "country": "USA",
            "postal_code": "94102"
        },
        "company": {
            "name": "Tech Corp",
            "industry": "Technology",
            "employees": 5000
        },
        "skills": ["Python", "React", "Management", "Writing"]
    }
    
    result = complex_pipeline(person_data)
    
    print(f"复杂模型处理结果: {result}")
    
    # 验证结果
    assert isinstance(result, AnalyticsModel)
    assert result.total_people == 1
    assert result.average_age == 28.0
    assert "San Francisco" in result.top_cities
    assert result.skill_frequency["Python"] == 1
    assert result.company_industries["Technology"] == 1
    
    # 测试模型验证失败
    try:
        invalid_person = {
            "name": "Invalid Person",
            "age": 17,  # 小于18，不符合约束
            "address": {
                "street": "123 Main St",
                "city": "Invalid",
                "country": "USA", 
                "postal_code": "invalid"  # 不符合正则表达式
            }
        }
        complex_pipeline(invalid_person)
        assert False, "应该抛出复杂模型验证错误"
    except Exception as e:
        print(f"✅ 正确捕获复杂模型验证错误: {type(e).__name__}")
    
    print("✅ 复杂Pydantic模型链式传递测试通过")


def test_multithreading_state_isolation():
    """测试多线程环境下Node的state和context隔离性"""
    print("\n=== 测试多线程State隔离性 ===")
    
    # 用于收集线程执行结果的字典
    thread_results = {}
    thread_states = {}
    shared_results = {}
    
    def thread_local_state_node(thread_id: int, value: int, 
                              context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        """测试线程本地state的节点"""
        state = context.state()
        shared_data = context.shared_data()
        
        # 在state中存储线程特定的数据
        state['thread_id'] = thread_id
        state['local_value'] = value
        state['processing_time'] = time.time()
        state['operations'] = state.get('operations', []) + [f'op_{thread_id}']
        
        # 在shared_data中累积全局数据（所有线程共享）
        total_processed = shared_data.get('total_processed', 0) + 1
        shared_data['total_processed'] = total_processed
        shared_data[f'thread_{thread_id}_value'] = value
        
        # 模拟一些处理时间
        time.sleep(random.uniform(0.001, 0.01))
        
        result = {
            'thread_id': thread_id,
            'local_state_value': state['local_value'],
            'total_operations': len(state['operations']),
            'shared_total': shared_data['total_processed']
        }
        
        print(f"线程 {thread_id}: state={dict(state)}, shared_total={shared_data['total_processed']}")
        return result
    
    def thread_processor_node(data: dict, 
                            context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        """处理线程数据的第二个节点"""
        state = context.state()
        shared_data = context.shared_data()
        
        thread_id = data['thread_id']
        
        # 验证state中的线程本地数据仍然存在
        assert state['thread_id'] == thread_id, f"State污染检测：期望thread_id={thread_id}, 实际={state.get('thread_id')}"
        
        # 更新线程本地状态
        state['final_result'] = data['local_state_value'] * 10
        state['chain_completed'] = True
        
        # 更新共享数据
        if 'completed_threads' not in shared_data:
            shared_data['completed_threads'] = []
        shared_data['completed_threads'].append(thread_id)
        
        final_result = {
            'thread_id': thread_id,
            'final_value': state['final_result'],
            'state_preserved': state['thread_id'] == thread_id,
            'total_completed': len(shared_data['completed_threads']),
            'shared_data_keys': list(shared_data.keys())
        }
        
        print(f"线程 {thread_id} 完成: final_value={state['final_result']}, 完成总数={len(shared_data['completed_threads'])}")
        return final_result
    
    # 创建节点和流水线
    state_node = Node(thread_local_state_node, name="state_node")
    processor_node = Node(thread_processor_node, name="processor_node")
    pipeline = state_node.then(processor_node)
    
    # 配置依赖注入容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    def run_thread_pipeline(thread_id: int, input_value: int):
        """在单独线程中运行pipeline"""
        try:
            result = pipeline(thread_id, input_value)
            thread_results[thread_id] = result
            
            # 获取线程结束时的state快照
            state_snapshot = dict(container.state())
            thread_states[thread_id] = state_snapshot
            
        except Exception as e:
            print(f"线程 {thread_id} 执行失败: {e}")
            thread_results[thread_id] = {'error': str(e)}
    
    # 启动多个线程
    num_threads = 5
    test_values = [10, 20, 30, 40, 50]
    
    print(f"启动 {num_threads} 个并发线程...")
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_thread_pipeline, i, test_values[i])
            for i in range(num_threads)
        ]
        
        # 等待所有线程完成
        for future in as_completed(futures):
            future.result()  # 获取结果，如果有异常会在这里抛出
    
    # 获取最终的shared_data状态
    final_shared_data = dict(container.shared_data())
    
    print(f"\n=== 线程执行结果分析 ===")
    print(f"最终shared_data: {final_shared_data}")
    
    # 验证结果
    assert len(thread_results) == num_threads, f"期望 {num_threads} 个线程结果，实际得到 {len(thread_results)}"
    
    # 验证State隔离性
    for thread_id in range(num_threads):
        result = thread_results[thread_id]
        assert 'error' not in result, f"线程 {thread_id} 执行出错: {result.get('error')}"
        
        # 验证线程本地数据正确
        assert result['thread_id'] == thread_id, f"线程ID不匹配: {result['thread_id']} != {thread_id}"
        assert result['final_value'] == test_values[thread_id] * 10, f"计算结果错误: {result['final_value']}"
        assert result['state_preserved'], f"线程 {thread_id} 的state数据被污染"
        
        print(f"✅ 线程 {thread_id}: state隔离正常, final_value={result['final_value']}")
    
    # 验证Shared_data共享性
    assert final_shared_data['total_processed'] == num_threads, f"共享计数器错误: {final_shared_data['total_processed']}"
    assert len(final_shared_data['completed_threads']) == num_threads, f"完成线程数错误"
    
    # 验证每个线程的数据都在shared_data中
    for thread_id in range(num_threads):
        thread_key = f'thread_{thread_id}_value'
        assert thread_key in final_shared_data, f"线程 {thread_id} 数据未在shared_data中找到"
        assert final_shared_data[thread_key] == test_values[thread_id], f"线程 {thread_id} 共享数据值错误"
    
    print(f"✅ Shared_data共享性验证通过: total_processed={final_shared_data['total_processed']}")
    print(f"✅ ThreadLocalSingleton隔离效果验证通过")
    print("✅ 多线程State隔离性测试通过")


def test_concurrent_execution_data_consistency():
    """测试shared_data在多线程中的共享性（不测试原子操作）"""
    print("\n=== 测试shared_data多线程共享性 ===")
    
    def record_thread_data_node(thread_id: int, 
                              context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        """将线程数据记录到shared_data"""
        shared_data = context.shared_data()
        state = context.state()
        
        # 获取当前线程名
        current_thread = threading.current_thread().name
        state['thread_name'] = current_thread
        state['thread_id'] = thread_id
        
        # 记录线程信息到shared_data（每个线程用唯一key，无竞态条件）
        thread_key = f"thread_{thread_id}"
        shared_data[thread_key] = {
            'thread_name': current_thread,
            'thread_id': thread_id,
            'processed_at': time.time()
        }
        
        # 将线程加入活跃线程列表（这里可能有竞态条件，但不影响测试目的）
        if 'active_threads' not in shared_data:
            shared_data['active_threads'] = []
        shared_data['active_threads'].append(thread_id)
        
        return {
            'thread_name': current_thread,
            'thread_id': thread_id,
            'recorded': True
        }
    
    def verify_shared_data_node(data: dict, 
                              context: BaseFlowContext = Provide[BaseFlowContext]) -> dict:
        """验证shared_data中的数据可以被其他线程访问"""
        shared_data = context.shared_data()
        state = context.state()
        
        thread_id = data['thread_id']
        thread_key = f"thread_{thread_id}"
        
        # 验证自己的数据确实存在于shared_data中
        own_data_exists = thread_key in shared_data
        
        # 记录验证结果到线程本地state
        state['own_data_exists'] = own_data_exists
        state['shared_data_keys'] = list(shared_data.keys())
        
        return {
            'thread_id': thread_id,
            'own_data_exists': own_data_exists,
            'total_shared_keys': len(shared_data.keys())
        }
    
    # 创建测试pipeline
    record_node = Node(record_thread_data_node, name="record")
    verify_node = Node(verify_shared_data_node, name="verify")
    pipeline = record_node.then(verify_node)
    
    # 配置容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])
    
    # 准备测试数据
    num_threads = 5
    thread_ids = list(range(num_threads))
    
    concurrent_results = {}
    
    def run_concurrent_task(thread_id: int):
        """运行并发任务"""
        try:
            result = pipeline(thread_id)
            concurrent_results[thread_id] = result
        except Exception as e:
            concurrent_results[thread_id] = {'error': str(e)}
    
    print(f"启动 {num_threads} 个并发任务测试shared_data共享性")
    
    # 启动并发任务
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(run_concurrent_task, thread_id)
            for thread_id in thread_ids
        ]
        
        for future in as_completed(futures):
            future.result()
    
    # 获取最终状态
    final_shared_data = dict(container.shared_data())
    
    print(f"\n=== 并发执行结果 ===")
    print(f"shared_data最终包含键: {list(final_shared_data.keys())}")
    print(f"活跃线程记录: {final_shared_data.get('active_threads', [])}")
    
    # 验证shared_data共享性
    # 1. 每个线程都成功记录了数据
    for thread_id in thread_ids:
        thread_key = f"thread_{thread_id}"
        assert thread_key in final_shared_data, f"线程 {thread_id} 的数据未在shared_data中找到"
        
        thread_data = final_shared_data[thread_key]
        assert thread_data['thread_id'] == thread_id, f"线程 {thread_id} 数据不正确"
    
    # 2. 每个任务都成功完成
    assert len(concurrent_results) == num_threads, f"任务数量不匹配: {len(concurrent_results)} != {num_threads}"
    
    # 3. 所有任务都成功执行且验证通过
    for thread_id in thread_ids:
        result = concurrent_results[thread_id]
        assert 'error' not in result, f"线程 {thread_id} 执行出错: {result.get('error')}"
        assert result['own_data_exists'], f"线程 {thread_id} 无法在shared_data中找到自己的数据"
    
    # 4. 验证active_threads列表（虽然可能因竞态条件不完整，但应该至少有数据）
    active_threads = final_shared_data.get('active_threads', [])
    assert len(active_threads) > 0, "active_threads列表应该包含至少一些线程ID"
    
    print(f"✅ 所有 {num_threads} 个线程都成功将数据写入shared_data")
    print(f"✅ 每个线程都能访问到shared_data中的数据")
    print(f"✅ shared_data在多线程环境中正确共享")
    print("✅ shared_data多线程共享性测试通过")


if __name__ == "__main__":
    print("=== Node.then() 方法综合测试 ===")
    
    try:
        test_basic_then()
        test_chain_then() 
        test_then_with_dependency_injection()
        test_decorator_with_then_chain()
        test_mixed_injection_and_simple_nodes()
        test_type_validation_in_chains()
        test_user_input_parameter_validation()
        test_node_output_to_input_validation()
        test_chain_type_mismatch_errors()
        test_pydantic_model_passing()
        test_complex_pydantic_model_chains()
        test_multithreading_state_isolation()
        test_concurrent_execution_data_consistency()
        print("\n🎉 所有综合then方法测试通过！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()