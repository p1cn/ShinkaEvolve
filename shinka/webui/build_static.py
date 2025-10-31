#!/usr/bin/env python3
"""
构建静态可视化网页脚本

将数据库数据注入到 viz_tree.html 中，生成一个完全独立的静态 HTML 文件。
不需要运行服务器，可以直接在浏览器中打开。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from shinka.database import DatabaseConfig, ProgramDatabase


def load_database_data(db_path: str) -> List[Dict[str, Any]]:
    """
    从数据库文件加载所有程序数据。
    
    Args:
        db_path: 数据库文件的绝对路径
        
    Returns:
        程序数据的字典列表
    """
    print(f"[INFO] 正在加载数据库: {db_path}")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    config = DatabaseConfig(db_path=db_path)
    db = ProgramDatabase(config, read_only=True)
    
    try:
        programs = db.get_all_programs()
        programs_dict = [p.to_dict() for p in programs]
        print(f"[INFO] 成功加载 {len(programs_dict)} 个程序")
        return programs_dict
    finally:
        db.close()


def inject_data_into_html(
    html_template_path: str, 
    programs_data: List[Dict[str, Any]], 
    output_path: str,
    db_name: str
) -> None:
    """
    将数据注入到 HTML 模板中，生成静态页面。
    
    Args:
        html_template_path: HTML 模板文件路径
        programs_data: 程序数据列表
        output_path: 输出文件路径
        db_name: 数据库名称（用于显示）
    """
    print(f"[INFO] 正在读取 HTML 模板: {html_template_path}")
    
    with open(html_template_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # 将数据序列化为 JSON
    programs_json = json.dumps(programs_data, ensure_ascii=False, indent=2)
    
    # 创建注入的脚本
    injection_script = f"""
    <script>
        // ========== 预加载的数据库数据 ==========
        // 这些数据在构建时从数据库中提取并注入到页面中
        const PRELOADED_DATABASE = {{
            name: {json.dumps(db_name, ensure_ascii=False)},
            path: "preloaded",
            programs: {programs_json}
        }};
        
        console.log("[STATIC BUILD] 使用预加载的数据库数据");
        console.log("[STATIC BUILD] 数据库名称:", PRELOADED_DATABASE.name);
        console.log("[STATIC BUILD] 程序数量:", PRELOADED_DATABASE.programs.length);
        
        // 覆盖原始的 loadDatabase 函数，使用预加载的数据
        window.addEventListener('DOMContentLoaded', function() {{
            console.log("[STATIC BUILD] DOM 加载完成，准备处理预加载数据");
            
            // 延迟一小段时间确保所有初始化代码都已执行
            setTimeout(function() {{
                if (typeof processData === 'function') {{
                    console.log("[STATIC BUILD] 调用 processData 处理预加载数据");
                    processData(PRELOADED_DATABASE.programs);
                }} else {{
                    console.error("[STATIC BUILD] processData 函数未定义");
                }}
            }}, 500);
        }});
    </script>
    """
    
    # 在 </head> 标签之前插入脚本
    if '</head>' in html_content:
        html_content = html_content.replace('</head>', f'{injection_script}\n</head>')
    else:
        print("[WARNING] 未找到 </head> 标签，将脚本添加到文件末尾")
        html_content += injection_script
    
    # 修改页面标题
    if '<title>' in html_content:
        original_title_start = html_content.find('<title>') + 7
        original_title_end = html_content.find('</title>')
        if original_title_end > original_title_start:
            new_title = f"{db_name} - ShinkaEvolve 可视化"
            html_content = (
                html_content[:original_title_start] + 
                new_title + 
                html_content[original_title_end:]
            )
    
    # 添加静态版本标记样式
    static_badge = """
    <style>
        .static-badge {
            position: fixed;
            top: 10px;
            right: 10px;
            background-color: #e74c3c;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            z-index: 10000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
    </style>
    """
    
    if '</head>' in html_content:
        head_pos = html_content.rfind('</head>')
        html_content = html_content[:head_pos] + static_badge + html_content[head_pos:]
    
    # 在 <body> 标签后添加徽章
    if '<body>' in html_content:
        body_start = html_content.find('<body>') + 6
        badge_html = '<div class="static-badge">📊 静态版本</div>\n'
        html_content = html_content[:body_start] + badge_html + html_content[body_start:]
    
    # 写入输出文件
    print(f"[INFO] 正在写入输出文件: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"[SUCCESS] ✅ 静态网页已生成: {output_path}")
    print(f"[INFO] 文件大小: {len(html_content) / 1024 / 1024:.2f} MB")


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="构建静态可视化网页，将数据库数据注入到 HTML 中"
    )
    parser.add_argument(
        "database",
        help="数据库文件路径（.db 或 .sqlite）"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 HTML 文件路径（默认：viz_static.html）",
        default=None
    )
    parser.add_argument(
        "-t", "--template",
        help="HTML 模板文件路径（默认：viz_tree.html）",
        default=None
    )
    parser.add_argument(
        "-n", "--name",
        help="数据库显示名称（默认：使用文件名）",
        default=None
    )
    
    args = parser.parse_args()
    
    # 解析路径
    db_path = os.path.abspath(args.database)
    
    # 确定模板路径
    if args.template:
        template_path = os.path.abspath(args.template)
    else:
        # 默认使用当前脚本所在目录的 viz_tree.html
        template_path = os.path.join(os.path.dirname(__file__), "viz_tree.html")
    
    # 确定输出路径
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        # 默认在数据库所在目录生成
        db_dir = os.path.dirname(db_path)
        output_path = os.path.join(db_dir, "viz_static.html")
    
    # 确定数据库名称
    if args.name:
        db_name = args.name
    else:
        db_name = os.path.splitext(os.path.basename(db_path))[0]
    
    try:
        # 加载数据
        programs_data = load_database_data(db_path)
        
        if not programs_data:
            print("[WARNING] ⚠️  数据库中没有程序数据")
            sys.exit(1)
        
        # 注入数据并生成静态页面
        inject_data_into_html(template_path, programs_data, output_path, db_name)
        
        print("\n" + "="*60)
        print("✅ 构建完成！")
        print("="*60)
        print(f"\n📄 输出文件: {output_path}")
        print(f"📊 数据库: {db_name}")
        print(f"🔢 程序数量: {len(programs_data)}")
        print(f"\n💡 使用方法: 直接在浏览器中打开 {os.path.basename(output_path)}")
        print("   或运行: open " + output_path)
        
    except FileNotFoundError as e:
        print(f"[ERROR] ❌ 文件未找到: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] ❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()







