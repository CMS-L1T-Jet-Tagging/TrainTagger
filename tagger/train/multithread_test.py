from joblib import Parallel, delayed, parallel_backend
import os
from math import sqrt
import time

def timeit(f):  
    """Measures execution time."""  
    def wrap(*args, **kwargs):  
        t1 = time.time()  
        res = f(*args, **kwargs)  
        print(f"{f.__name__} ran in {time.time() - t1:.6f}s")  
        #return res  
    return wrap  

os.system('nproc')

@timeit
def parallel_in_parallel_wrapper(n=10,n_threads=2,backend='loky'):

    
    if backend!='loky':
        with parallel_backend(backend):
            intermediate_out = Parallel(n_jobs=n_threads)(delayed(sqrt)(i ** 2) for i in range(n))

    else:
        with parallel_backend(backend,inner_max_num_threads=2):
             intermediate_out = Parallel(n_jobs=n_threads)(delayed(sqrt)(i ** 2) for i in range(n))
    #return intermediate_out


threads = [1,2,4,8,12,24]
n_iterations = [10000,1000000]

for thread in threads:
    for iterations in n_iterations:
        print("=========================")
        print("n threads: ",thread)
        print("n iterations: ",iterations)
        print("+++++++++++++")
        print("single thread 2 backend thread loky: ")
        parallel_in_parallel_wrapper(iterations,n_threads=thread,backend='loky')
        print("+++++++++++++")
        print("multi thread 2 backend thread loky: ")
        parallel_in_parallel_wrapper(iterations,n_threads=thread,backend='loky')
        print("+++++++++++++")
        print("single thread sequential: ")
        parallel_in_parallel_wrapper(iterations,n_threads=thread,backend='sequential')

        print("=========================")
