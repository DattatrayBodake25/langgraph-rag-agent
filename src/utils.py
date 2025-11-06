import datetime
import sys

# ANSI color codes for clean console output
class LogColors:
    INFO = "\033[94m"      # Blue
    SUCCESS = "\033[92m"   # Green
    WARNING = "\033[93m"   # Yellow
    ERROR = "\033[91m"     # Red
    RESET = "\033[0m"      # Reset to default color

def log(message: str, level: str = "INFO"):
    """
    Simple color-coded logger with timestamp and severity levels.
    Usage:
        log("Loaded documents.", "INFO")
        log("No results found.", "WARNING")
        log("Answer generated.", "SUCCESS")
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = level.upper()

    color_map = {
        "INFO": LogColors.INFO,
        "SUCCESS": LogColors.SUCCESS,
        "WARNING": LogColors.WARNING,
        "ERROR": LogColors.ERROR,
    }
    color = color_map.get(level, LogColors.INFO)

    formatted_message = f"[{timestamp}] [{level}] {message}"
    print(f"{color}{formatted_message}{LogColors.RESET}", file=sys.stdout)