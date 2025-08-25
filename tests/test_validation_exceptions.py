#!/usr/bin/env python3
"""
test_validation_exceptions.py - 测试custom_validate_call包装器的验证异常分类功能

验证输入参数异常和输出返回值异常是否能正确分类为：
- ValidationInputException: 输入参数验证失败
- ValidationOutputException: 返回值验证失败
"""

import pytest
from pydantic import BaseModel, Field

from src.aetherflow import (
    BaseFlowContext,
    ValidationInputException,
    ValidationOutputException,
    custom_validate_call,
    node,
)


# 测试用的Pydantic模型
class UserModel(BaseModel):
    id: int = Field(gt=0, description="用户ID必须大于0")
    name: str = Field(min_length=1, max_length=50, description="用户名长度1-50")
    email: str = Field(pattern=r"^[^@]+@[^@]+\.[^@]+$", description="有效邮箱格式")
    age: int | None = Field(None, ge=0, le=150, description="年龄0-150")


class ProductModel(BaseModel):
    name: str
    price: float = Field(gt=0, description="价格必须大于0")
    tags: list[str] = Field(default_factory=list)


class OrderModel(BaseModel):
    user: UserModel
    products: list[ProductModel]
    total: float = Field(gt=0)


class TestValidationExceptions:
    """测试验证异常分类"""

    @pytest.fixture(autouse=True)
    def setup_injection(self, wired_container):
        """自动设置依赖注入"""
        self.container = wired_container(__name__)
        yield
        self.container.unwire()

    def test_validation_input_exception(self):
        """测试输入参数验证异常"""

        @node(enable_retry=False)
        def strict_input_node(x: int) -> str:
            return f"processed: {x}"

        with pytest.raises(ValidationInputException) as exc_info:
            strict_input_node("not_int")  # 传入字符串而不是int

        exception = exc_info.value
        assert "输入参数验证失败" in str(exception)
        assert exception.node_name == "strict_input_node"
        assert exception.retryable == False

    def test_validation_output_exception(self):
        """测试输出返回值验证异常"""

        @node(enable_retry=False)
        def strict_output_node(x: int) -> int:
            return f"string_instead_of_int_{x}"  # 返回字符串而不是int

        with pytest.raises(ValidationOutputException) as exc_info:
            strict_output_node(42)

        exception = exc_info.value
        assert "返回值验证失败" in str(exception)
        assert exception.node_name == "strict_output_node"
        assert exception.retryable == False

    def test_custom_validate_call_direct_usage(self):
        """测试直接使用custom_validate_call装饰器"""

        @custom_validate_call(validate_return=True, node_name="direct_test")
        def direct_test_func(x: int) -> int:
            return f"wrong_type_{x}"  # 返回错误类型

        # 输入验证异常
        with pytest.raises(ValidationInputException):
            direct_test_func("not_int")

        # 输出验证异常
        with pytest.raises(ValidationOutputException):
            direct_test_func(42)

    def test_pydantic_model_input_validation(self):
        """测试Pydantic模型作为输入参数的验证"""

        @custom_validate_call(validate_return=True, node_name="pydantic_input")
        def create_user_summary(user: UserModel) -> str:
            return f"用户{user.name}({user.id})，邮箱：{user.email}"

        # 正确的用户模型
        valid_user = UserModel(id=1, name="张三", email="zhangsan@example.com", age=25)
        result = create_user_summary(valid_user)
        assert result == "用户张三(1)，邮箱：zhangsan@example.com"

        # 错误的输入参数（无效的用户数据）
        with pytest.raises(ValidationInputException) as exc_info:
            create_user_summary(
                {
                    "id": -1,  # 无效ID
                    "name": "",  # 空名称
                    "email": "invalid-email",  # 无效邮箱
                    "age": 200,  # 无效年龄
                }
            )

        assert "输入参数验证失败" in str(exc_info.value)

    def test_pydantic_model_output_validation(self):
        """测试Pydantic模型作为返回值的验证"""

        @custom_validate_call(validate_return=True, node_name="pydantic_output")
        def create_invalid_user(user_id: int, name: str) -> UserModel:
            # 故意返回无效的用户数据
            return {
                "id": -1,  # 无效ID
                "name": "",  # 空名称
                "email": "not-an-email",  # 无效邮箱
                "age": 300,  # 无效年龄
            }

        with pytest.raises(ValidationOutputException) as exc_info:
            create_invalid_user(1, "test")

        assert "返回值验证失败" in str(exc_info.value)

    def test_complex_nested_model_validation(self):
        """测试复杂嵌套模型的验证"""

        @custom_validate_call(validate_return=True, node_name="nested_model")
        def create_order(user_data: dict, product_list: list[dict]) -> OrderModel:
            user = UserModel(**user_data)
            products = [ProductModel(**p) for p in product_list]
            total = sum(p.price for p in products)

            return OrderModel(user=user, products=products, total=total)

        # 正确的数据
        valid_user_data = {"id": 1, "name": "测试用户", "email": "test@example.com"}
        valid_products = [
            {"name": "商品1", "price": 100.0, "tags": ["tag1"]},
            {"name": "商品2", "price": 200.0, "tags": ["tag2"]},
        ]

        result = create_order(valid_user_data, valid_products)
        assert isinstance(result, OrderModel)
        assert result.total == 300.0
        assert len(result.products) == 2

        # 输入验证失败 - 无效用户数据
        with pytest.raises(ValidationInputException):
            create_order(
                {"id": "not_int", "name": "", "email": "invalid"},  # 无效用户
                valid_products,
            )

    def test_complex_output_validation_failure(self):
        """测试复杂输出验证失败"""

        @custom_validate_call(validate_return=True, node_name="complex_output_fail")
        def create_broken_order() -> OrderModel:
            # 返回不符合OrderModel要求的数据
            return {
                "user": {"id": -1, "name": "", "email": "bad"},  # 无效用户
                "products": [{"name": "test", "price": -100}],  # 负价格
                "total": -500,  # 负总价
            }

        with pytest.raises(ValidationOutputException) as exc_info:
            create_broken_order()

        exception = exc_info.value
        assert "返回值验证失败" in str(exception)
        assert exception.node_name == "complex_output_fail"


if __name__ == "__main__":
    print("=== ValidationExceptions 测试 ===")

    # 配置依赖注入容器
    container = BaseFlowContext()
    container.wire(modules=[__name__])

    try:
        test_instance = TestValidationExceptions()

        test_instance.test_validation_input_exception()
        test_instance.test_validation_output_exception()
        test_instance.test_validation_input_complex_types()
        test_instance.test_validation_output_complex_types()
        test_instance.test_successful_validation()
        test_instance.test_custom_validate_call_direct_usage()
        test_instance.test_exception_inheritance_and_attributes()

        print("\n🎉 所有验证异常分类测试通过！")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()

    finally:
        container.unwire()
