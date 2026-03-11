"""AFMLKit Web UI 运行入口

使用方法:
    streamlit run webapp/__main__.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    """启动 Streamlit 应用"""
    app_path = Path(__file__).parent / "app.py"

    if not app_path.exists():
        print(f"错误：找不到应用文件 {app_path}")
        sys.exit(1)

    print(f"启动 AFMLKit Web UI...")
    print(f"应用路径：{app_path}")
    print(f"访问地址：http://localhost:8501")
    print()

    # 运行 streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])


if __name__ == "__main__":
    main()
