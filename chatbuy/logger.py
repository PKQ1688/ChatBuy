import logging
import sys
from rich.logging import RichHandler

# 配置日志记录器
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)

# 获取日志记录器实例
log = logging.getLogger("rich")

# 示例用法 (可选，用于测试)
if __name__ == "__main__":
    log.info("这是一个信息日志")
    log.warning("这是一个警告日志")
    log.error("这是一个错误日志")
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("这是一个异常日志")