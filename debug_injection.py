#!/usr/bin/env python3
"""
调试依赖注入问题的精确测试脚本
"""

from dependency_injector.wiring import Provide
from src.aetherflow import BaseFlowContext, node
from tests.shared.data_models import StandardTestData, create_test_data
from tests.fixtures.injection_helpers import setup_test_container

# 模拟测试中的确切情况
def test_exact_scenario():
    """精确模拟测试场景"""
    print("=== 模拟测试场景 ===")
    
    # 使用与测试相同的容器设置
    print(f"Module name: {__name__}")
    test_container = setup_test_container(__name__)
    container = test_container.container
    
    print(f"Test container type: {type(test_container)}")
    print(f"Container type: {type(container)}")
    print(f"Container providers: {list(container.providers.keys())}")
    print(f"Wired modules: {test_container._wired_modules}")
    
    # 检查容器是否正确 wired
    try:
        wiring_config = getattr(container, '_wiring_config', None)
        print(f"Container wiring config: {wiring_config}")
    except Exception as e:
        print(f"Error checking wiring: {e}")
    
    # 先测试直接使用容器实例
    @node
    def injection_node_direct(data: StandardTestData, 
                             context: BaseFlowContext = Provide[container.__self__]) -> StandardTestData:
        print(f"Direct injection - context type: {type(context)}")
        print(f"Direct injection - context: {context}")
        print(f"Direct injection - context has state: {hasattr(context, 'state')}")
        
        if hasattr(context, 'state'):
            state = context.state()
            print(f"Direct injection - state: {state}")
            state['input_value'] = data.value
            state['processing_step'] = 'direct_injection_test'
        else:
            print("ERROR: Context has no 'state' attribute!")
            print(f"Context attributes: {dir(context)}")
            raise AttributeError("'Provide' object has no attribute 'state'")
        
        return StandardTestData(
            value=data.value + "_direct_processed",
            metadata={"processed_by": "direct_injection_node"}
        )
    
    @node
    def injection_node(data: StandardTestData, 
                      context: BaseFlowContext = Provide["<container>"]) -> StandardTestData:
        print(f"Inside injection_node - context type: {type(context)}")
        print(f"Inside injection_node - context: {context}")
        print(f"Inside injection_node - context has state: {hasattr(context, 'state')}")
        
        if hasattr(context, 'state'):
            state = context.state()
            print(f"Inside injection_node - state: {state}")
            state['input_value'] = data.value
            state['processing_step'] = 'injection_test'
        else:
            print("ERROR: Context has no 'state' attribute!")
            print(f"Context attributes: {dir(context)}")
            raise AttributeError("'Provide' object has no attribute 'state'")
        
        return StandardTestData(
            value=data.value * 2,
            name=f"injected_{data.name}",
            metadata={"injected": True}
        )
    
    # 执行测试
    print("\nExecuting injection_node...")
    try:
        test_data = create_test_data(10, "injection_input")
        result = injection_node(test_data)
        print(f"Success! Result: {result}")
        
        # 验证状态
        final_state = test_container.get_state()
        print(f"Final state: {final_state}")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_scenario()