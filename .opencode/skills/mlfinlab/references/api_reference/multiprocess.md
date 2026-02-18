# API Reference: multiprocess.py

**Language**: Python

**Source**: `util/multiprocess.py`

---

## Functions

### lin_parts(num_atoms, num_threads)

Advances in Financial Machine Learning, Snippet 20.5, page 306.

The lin_parts function

The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
where the number of subsets is the minimum between the number of processors and the number
of atoms. For N subsets we need to find the N+1 indices that enclose the partitions.
This logic is demonstrated in Snippet 20.5.

This function partitions a list of atoms in subsets (molecules) of equal size.
An atom is a set of indivisible set of tasks.

:param num_atoms: (int) Number of atoms
:param num_threads: (int) Number of processors
:return: (np.array) Partition of atoms

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_atoms | None | - | - |
| num_threads | None | - | - |

**Returns**: (none)



### nested_parts(num_atoms, num_threads, upper_triangle = False)

Advances in Financial Machine Learning, Snippet 20.6, page 308.

The nested_parts function

This function enables parallelization of nested loops.
:param num_atoms: (int) Number of atoms
:param num_threads: (int) Number of processors
:param upper_triangle: (bool) Flag to order atoms as an upper triangular matrix (including the main diagonal)
:return: (np.array) Partition of atoms

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| num_atoms | None | - | - |
| num_threads | None | - | - |
| upper_triangle | None | False | - |

**Returns**: (none)



### mp_pandas_obj(func, pd_obj, num_threads = 24, mp_batches = 1, lin_mols = True, verbose = True)

Advances in Financial Machine Learning, Snippet 20.7, page 310.

The mpPandasObj, used at various points in the book

Parallelize jobs, return a dataframe or series.
Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)

First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
cores will be idle half of the runtime, waiting for the first core to process molecule 1.
Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
every core will receive equal workload, even though the first 10 molecules take as much time as the
next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
mpBatches=1.

Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
the molecule.

Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
Once the code is debugged, we will want to use numThreads>1.

Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.

:param func: (function) A callback function, which will be executed in parallel
:param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
:param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
:param mp_batches: (int) Number of parallel batches (jobs per core)
:param lin_mols: (bool) Tells if the method should use linear or nested partitioning
:param verbose: (bool) Flag to report progress on asynch jobs
:param kargs: (var args) Keyword arguments needed by func
:return: (pd.DataFrame) of results

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| func | None | - | - |
| pd_obj | None | - | - |
| num_threads | None | 24 | - |
| mp_batches | None | 1 | - |
| lin_mols | None | True | - |
| verbose | None | True | - |

**Returns**: (none)



### process_jobs_(jobs)

Advances in Financial Machine Learning, Snippet 20.8, page 311.

Single thread execution, for debugging

Run jobs sequentially, for debugging

:param jobs: (list) Jobs (molecules)
:return: (list) Results of jobs

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| jobs | None | - | - |

**Returns**: (none)



### expand_call(kargs)

Advances in Financial Machine Learning, Snippet 20.10.

Passing the job (molecule) to the callback function

Expand the arguments of a callback function, kargs['func']

:param kargs: Job (molecule)
:return: Result of a job

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| kargs | None | - | - |

**Returns**: (none)



### report_progress(job_num, num_jobs, time0, task)

Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.

Example of Asynchronous call to pythons multiprocessing library

:param job_num: (int) Number of current job
:param num_jobs: (int) Total number of jobs
:param time0: (time) Start time
:param task: (str) Task description
:return: (None)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| job_num | None | - | - |
| num_jobs | None | - | - |
| time0 | None | - | - |
| task | None | - | - |

**Returns**: (none)



### process_jobs(jobs, task = None, num_threads = 24, verbose = True)

Advances in Financial Machine Learning, Snippet 20.9.2, page 312.

Example of Asynchronous call to pythons multiprocessing library

Run in parallel. jobs must contain a 'func' callback, for expand_call

:param jobs: (list) Jobs (molecules)
:param task: (str) Task description
:param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
:param verbose: (bool) Flag to report progress on asynch jobs
:return: (None)

**Parameters**:

| Name | Type | Default | Description |
|------|------|---------|-------------|
| jobs | None | - | - |
| task | None | None | - |
| num_threads | None | 24 | - |
| verbose | None | True | - |

**Returns**: (none)


