---

## 1. 创建下载器模块

- [x] 1.1 创建 `src/binance_downloader.py`
- [x] 1.2 实现 `argparse` 命令行参数解析（--start, --end, --backfill, --help）
- [x] 1.3 实现日期范围生成函数
- [x] 1.4 实现下载函数（带重试逻辑）
- [x] 1.5 实现断点续传检测（检查已存在文件）
- [x] 1.6 实现增量回补逻辑（--backfill 模式）

## 2. 测试

- [x] 2.1 测试帮助信息 `python src/binance_downloader.py --help`
- [x] 2.2 测试下载功能（指定小日期范围）
- [x] 2.3 测试断点续传（重复运行应跳过已存在文件）
- [x] 2.4 测试回补功能（删除某个文件后重新运行）

---

Ready to implement?
