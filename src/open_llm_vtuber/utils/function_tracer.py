import time
import asyncio
import functools
import logging
import os
from pathlib import Path
from typing import Union, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

def get_log_file():
    """Get the log file path, create directory if not exists"""
    log_dir = Path("logs/function_traces")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    return log_dir / f"function_traces_{current_date}.log"

def write_to_log(message: str):
    """Write message to log file"""
    try:
        log_file = get_log_file()
        with open(log_file, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"{timestamp} | {message}\n")
    except Exception as e:
        logger.error(f"Failed to write to trace log: {e}")

def trace_calls(func: Union[Callable, None] = None, *, print_time: bool = True) -> Callable:
    """
    A simple decorator that measures and prints function execution time.
    Supports both synchronous and asynchronous functions.
    
    Args:
        func: The function to be decorated
        print_time: Whether to print the execution time
    
    Usage:
        @trace_calls
        def sync_function():
            pass

        @trace_calls
        async def async_function():
            pass

        # Or with parameters
        @trace_calls(print_time=False)
        async def silent_function():
            pass
    """
    def decorator(fn: Callable) -> Callable:
        is_async = asyncio.iscoroutinefunction(fn)
        
        if is_async:
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    if print_time:
                        message = f"{fn.__name__}: {duration_ms:.3f}ms"
                        logger.info(message)
                        write_to_log(message)
            
            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration_ms = (end_time - start_time) * 1000
                    if print_time:
                        message = f"{fn.__name__}: {duration_ms:.3f}ms"
                        logger.info(message)
                        write_to_log(message)
            
            return sync_wrapper

    if func is None:
        return decorator
    return decorator(func) 