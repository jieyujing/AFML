#!/usr/bin/env python
"""
AFMLKit Web UI 快速启动脚本

使用方法:
    python run_webapp.py

或者直接使用:
    streamlit run webapp/app.py
"""
import subprocess
import sys
import os
from pathlib import Path


def check_dependencies():
    """检查必要的依赖是否已安装"""
    required = ['streamlit', 'plotly', 'yaml', 'pandas', 'numpy']
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("⚠️ 缺少以下依赖:")
        for pkg in missing:
            print(f"   - {pkg}")
        print()
        print("请运行以下命令安装:")
        print("   pip install -e \".[webapp]\"")
        print()
        return False

    return True


def main():
    """主函数"""
    print("=" * 60)
    print("  AFMLKit Web UI")
    print("  金融机器学习工具包 - 研究版")
    print("=" * 60)
    print()

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 获取应用路径
    app_path = Path(__file__).parent / "webapp" / "app.py"

    if not app_path.exists():
        print(f"❌ 错误：找不到应用文件 {app_path}")
        sys.exit(1)

    # 设置环境变量
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

    # 启动 Streamlit
    print("🚀 启动 Web UI...")
    print()
    print("访问地址：http://localhost:8501")
    print()
    print("按 Ctrl+C 停止应用")
    print()
    print("-" * 60)
    print()

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print()
        print()
        print("👋 应用已停止")
    except Exception as e:
        print(f"❌ 错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
