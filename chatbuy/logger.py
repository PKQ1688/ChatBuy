import logging

from rich.logging import RichHandler

# Configure logger
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=True, markup=True)],
)

log = logging.getLogger("chatbuy")

if __name__ == "__main__":
    log.info("这是一个信息日志")
    log.warning("这是一个警告日志")
    log.error("这是一个错误日志")
    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("这是一个异常日志")
