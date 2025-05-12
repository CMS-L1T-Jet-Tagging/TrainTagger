#!/usr/bin/env python3
# Import necessary modules
import sys
import sysconfig
import math
import time
from threading import Thread
from multiprocessing import Process

# Define a decorator to measure function execution time
def time_taken(func):
    """
    A decorator to measure the execution time of a function.
    
    Args:
        func: The target function.
    
    Returns:
        A wrapper function that measures and prints the execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Function {func.__name__!r} took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper

# Define a compute-intensive task function
def compute_intensive_task(num):
    """
    A compute-intensive task that calculates the factorial of a number.
    
    Args:
        num: The input number.
    
    Returns:
        The factorial of the input number.
    """
    #return math.sqrt(num)  # This line is commented out to focus on threading
    return math.factorial(num)

# Define single-threaded task function
@time_taken
def single_threaded_task(nums):
    """
    A single-threaded task that performs compute-intensive tasks sequentially.
    
    Args:
        nums: A list of input numbers.
    """
    for num in nums:
        compute_intensive_task(num)

# Define multi-threaded task function
@time_taken
def multi_threaded_task(nums):
    """
    A multi-threaded task that creates and runs a thread pool to perform 
    compute-intensive tasks concurrently.
    
    Args:
        nums: A list of input numbers.
    """
    threads = []

    # Create len(nums) threads
    for num in nums:
        thread = Thread(target=compute_intensive_task, args=(num,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Define multi-processing task function
@time_taken
def multi_processing_task(nums):
    """
    A multi-processing task that creates and runs a process pool to perform 
    compute-intensive tasks concurrently.
    
    Args:
        nums: A list of input numbers.
    """
    processes = []

    # Create len(nums) processes
    for num in nums:
        process = Process(target=compute_intensive_task, args=(num,))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()

# Define the main function
def main():
    """
    The main entry point of the program.
    
    It prints the Python version and checks if the GIL is enabled or not.
    Then, it runs single-threaded, multi-threaded, and multi-processing tasks.
    """
    print(f"Python Version: {sys.version}")

    # Check GIL status
    py_version = float(".".join(sys.version.split()[0].split(".")[0:2]))
    status = sysconfig.get_config_var("Py_GIL_DISABLED")

    if py_version >= 3.13:
        status = sys._is_gil_enabled()
    if status is None:
        print("GIL cannot be disabled for Python version <= 3.12")
    if status == 0:
        print("GIL is currently disabled")
    if status == 1:
        print("GIL is currently active")

    threads = [1,2,4,8,12,24,96,384]
    for thread in threads:
        print("===================")
        print("Num threads: ",thread)
        nums = [300001 for i in range(thread)]

        # Run single-threaded task
        single_threaded_task(nums)

        # Run multi-threaded task
        multi_threaded_task(nums)

        # Run multi-processing task
        multi_processing_task(nums)

# Call the main function
if __name__ == "__main__":
    main()