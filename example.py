#!/usr/bin/env python3
"""
AetherFlow 流式接口示例

这个示例展示了如何使用 AetherFlow 的流式接口构建复杂的工作流。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from aetherflow import node, LoopControl, AppContainer

# ===== 基础节点定义 =====

@node
def get_user_input(query: str) -> dict:
    """获取用户输入并解析"""
    print(f"Processing user query: '{query}'")
    return {
        "topic": "machine learning",
        "complexity": "intermediate",
        "format": "tutorial"
    }

@node
def research_basics(topic: str) -> dict:
    """研究基础概念"""
    print(f"Researching basics of {topic}...")
    return {"basic_concepts": f"Basic concepts of {topic}"}

@node
def research_advanced(topic: str) -> dict:
    """研究高级概念"""
    print(f"Researching advanced aspects of {topic}...")
    return {"advanced_concepts": f"Advanced concepts of {topic}"}

@node
def combine_research(parallel_results: dict) -> dict:
    """合并研究结果"""
    print("Combining research results...")
    basics = parallel_results.get('basic_concepts', '')
    advanced = parallel_results.get('advanced_concepts', '')
    return {"research_summary": f"{basics} + {advanced}"}

@node
def check_complexity(complexity: str) -> bool:
    """检查复杂度是否为高级"""
    result = complexity == "advanced"
    print(f"Is complexity '{complexity}' advanced? {result}")
    return result

@node
def create_basic_content(research_summary: str, format: str) -> dict:
    """创建基础内容"""
    print(f"Creating basic {format} content...")
    return {"content": f"Basic {format}: {research_summary}"}

@node
def create_advanced_content(research_summary: str, format: str) -> dict:
    """创建高级内容"""
    print(f"Creating advanced {format} content...")
    return {"content": f"Advanced {format}: {research_summary}"}

@node
def refine_content(content: str, iteration: int = 0) -> dict:
    """优化内容，最多3次迭代"""
    print(f"Refining content (iteration {iteration + 1})...")
    
    if iteration >= 2:  # 最多3次迭代
        print("Content refinement complete!")
        return LoopControl.BREAK
    
    refined = content + f" [refined-{iteration + 1}]"
    return {
        "content": refined,
        "iteration": iteration + 1
    }

@node
def finalize_output(content: str, format: str) -> dict:
    """最终输出"""
    print(f"Finalizing {format} output...")
    return {"final_output": f"FINAL {format.upper()}: {content}"}

def main():
    """主函数 - 演示各种工作流模式"""
    
    print("=" * 60)
    print("AetherFlow 流式接口完整示例")
    print("=" * 60)
    
    # 构建复杂的工作流：
    # 1. 获取用户输入
    # 2. 并行研究基础和高级概念
    # 3. 合并研究结果
    # 4. 根据复杂度选择内容创建策略
    # 5. 循环优化内容
    # 6. 最终输出
    
    workflow = (
        get_user_input
        .fan_out_to([research_basics, research_advanced])
        .fan_in(combine_research)
        .then(check_complexity.branch_on({
            True: create_advanced_content,
            False: create_basic_content
        }))
        .then(refine_content.repeat(5))  # 最多5次，但会在3次后自动退出
        .then(finalize_output)
    )
    
    print("\n🚀 执行工作流...\n")
    
    # 执行工作流
    result = workflow.run({
        "query": "How do I learn machine learning effectively?"
    })
    
    print("\n" + "=" * 60)
    print("📋 最终结果:")
    print("=" * 60)
    
    for key, value in result.items():
        if key == 'final_output':
            print(f"\n🎯 {key}: {value}")
        else:
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ 工作流执行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()