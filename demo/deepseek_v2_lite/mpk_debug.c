// mpk_debug.c — host-side watchdog + runtime-protocol helpers for Mirage's generated launcher.
// run_layer.py::splice_hardcode() puts this text into mirage.mpk.persistent_kernel.HARD_CODE before
// compile(), so it lands in the same translation unit as persistent_kernel.cuh (global_runtime_config and
// launch_persistent_kernel are visible). The ref clone itself is not modified.
//   launch_async_func(stream)     launch_persistent_kernel on a detached thread. Fleet's launch_func holds the
//                                 GIL through the blocking cudaStreamSynchronize, so Python polls launch_poll_func.
//   launch_poll_func()            -1.0 while the launch runs, else its wall ms.
//   set_max_seq_length_func(n)    RUNTIME termination bound of prepare_next_batch (step + 2 >= max_seq_length).
//                                 Before this it was only settable by a second init_func, which stalls the next
//                                 launch (docs/gpu-log.md session 4).
//   reset_event_timing_func()     event_timing_count = 0 (before: only init_func reset it).
//   debug_state_func([timeout_s]) dict of the runtime's host-visible state (event counters, num_triggers, queue
//                                 heads, step, request slots, event-timing tail), each array copied over a side
//                                 stream with a timeout so a hung kernel cannot hang the dump.
#include <atomic>
#include <chrono>
#include <thread>

static std::atomic<int> g_launch_state{0};  // 0 idle, 1 running, 2 done
static double g_launch_ms = 0.0;

static PyObject *launch_async_func(PyObject *self, PyObject *args) {
  PyObject *py_stream;
  if (!PyArg_ParseTuple(args, "O", &py_stream)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  if (g_launch_state == 1) {
    PyErr_SetString(PyExc_RuntimeError, "a launch is still running");
    return NULL;
  }
  cudaStream_t stream = (cudaStream_t)PyLong_AsVoidPtr(py_stream);
  g_launch_state = 1;
  std::thread([stream]() {
    auto t0 = std::chrono::steady_clock::now();
    launch_persistent_kernel(stream);
    g_launch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    g_launch_state = 2;
  }).detach();
  Py_RETURN_NONE;
}

static PyObject *launch_poll_func(PyObject *self, PyObject *args) {
  return PyFloat_FromDouble(g_launch_state == 2 ? g_launch_ms : -1.0);
}

static PyObject *set_max_seq_length_func(PyObject *self, PyObject *args) {
  int n;
  if (!PyArg_ParseTuple(args, "i", &n)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  global_runtime_config.max_seq_length = n;
  Py_RETURN_NONE;
}

static PyObject *reset_event_timing_func(PyObject *self, PyObject *args) {
  if (global_runtime_config.event_timing_count != nullptr) {
    (void)hipMemset(global_runtime_config.event_timing_count, 0, sizeof(int));
  }
  Py_RETURN_NONE;
}

// ---- debug state (read while the kernel may be spinning) --------------------------------------------------
static char *g_dbg_host = nullptr;         // pinned scratch, one array at a time
static size_t const DBG_BYTES = 4u << 20;  // largest array: 8 XCDs x events x 8 B, or the event-timing tail

static bool dbg_copy(void const *src, size_t bytes, hipStream_t s, double timeout_s) {
  if (src == nullptr || bytes == 0 || bytes > DBG_BYTES) {
    return false;
  }
  if (hipMemcpyAsync(g_dbg_host, src, bytes, hipMemcpyDeviceToHost, s) != hipSuccess) {
    return false;
  }
  auto t0 = std::chrono::steady_clock::now();
  while (hipStreamQuery(s) != hipSuccess) {  // D2H goes through SDMA, not the CUs the persistent kernels hold
    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

// list of n numbers read from device memory (elem 4 = int, 8 = unsigned 64-bit); None if unreadable
static PyObject *dbg_list(void const *src, int n, int elem, hipStream_t s, double timeout_s, int *fails) {
  if (src == nullptr || n <= 0) {
    Py_RETURN_NONE;
  }
  if (!dbg_copy(src, (size_t)n * elem, s, timeout_s)) {
    (*fails)++;
    Py_RETURN_NONE;
  }
  PyObject *l = PyList_New(n);
  for (int i = 0; i < n; i++) {
    PyList_SET_ITEM(l, i, elem == 8 ? PyLong_FromUnsignedLongLong(((unsigned long long const *)g_dbg_host)[i])
                                    : PyLong_FromLong(((int const *)g_dbg_host)[i]));
  }
  return l;
}

static PyObject *debug_state_func(PyObject *self, PyObject *args) {
  double timeout_s = 5.0;
  if (!PyArg_ParseTuple(args, "|d", &timeout_s)) {
    PyErr_SetString(PyExc_TypeError, "Invalid parameters");
    return NULL;
  }
  RuntimeConfig const &c = global_runtime_config;
  if (g_dbg_host == nullptr && hipHostMalloc((void **)&g_dbg_host, DBG_BYTES, hipHostMallocDefault) != hipSuccess) {
    PyErr_SetString(PyExc_RuntimeError, "hipHostMalloc failed");
    return NULL;
  }
  hipStream_t s;  // never destroyed: a timed-out copy may still be pending on it
  (void)hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
  int fails = 0;
  int const ne = c.num_events, nw = c.num_workers, ns = c.num_local_schedulers + c.num_remote_schedulers;
  PyObject *d = PyDict_New();
#define DBG_INT(key, v) PyDict_SetItemString(d, key, PyLong_FromLongLong((long long)(v)))
#define DBG_ARR(key, ptr, n, elem) PyDict_SetItemString(d, key, dbg_list(ptr, n, elem, s, timeout_s, &fails))
  DBG_INT("num_events", ne);
  DBG_INT("num_workers", nw);
  DBG_INT("num_schedulers", ns);
  DBG_INT("num_xcds", c.num_xcds);
  DBG_INT("max_seq_length", c.max_seq_length);
  DBG_INT("eos_token_id", c.eos_token_id);
  DBG_INT("total_num_requests", c.total_num_requests);
  DBG_INT("launch_state", g_launch_state.load());
  DBG_ARR("event_counters", c.all_event_counters, ne, 8);
  DBG_ARR("num_triggers", c.all_event_num_triggers, ne, 4);
  DBG_ARR("xcd_local_event_counters", c.xcd_local_event_counters, c.num_xcds * ne, 8);
  DBG_ARR("xcd_event_num_tasks", c.xcd_event_num_tasks, c.num_xcds * ne, 4);
  DBG_ARR("sched_last_ready", c.sched_queue_last_ready_event_id, ns + 1, 8);
  DBG_ARR("sched_next_free", c.sched_queue_next_free_event_id, ns + 1, 8);
  DBG_ARR("worker_last_ready", c.worker_queue_last_ready_task_id, 2 * nw, 8);
  DBG_ARR("worker_xcd_ready_count", c.worker_xcd_ready_count, 1, 4);
  DBG_ARR("step", c.step, c.total_num_requests, 4);
  DBG_ARR("request_ids", c.request_ids, MPK_MAX_NUM_BATCHED_REQUESTS, 4);
  DBG_ARR("next_request_id", c.next_request_id, 1, 4);
  DBG_ARR("page_queue_head", c.page_queue_head, 1, 4);
  DBG_ARR("page_queue_tail", c.page_queue_tail, 1, 4);
  DBG_ARR("qo_indptr", c.qo_indptr_buffer, MPK_MAX_NUM_BATCHED_REQUESTS + 1, 4);
  DBG_ARR("input_tokens", c.input_tokens, 1, 8);
  DBG_ARR("output_tokens", c.output_tokens, 1, 8);
  DBG_ARR("event_timing_count", c.event_timing_count, 1, 4);
  // the last 32 (event index, s_memrealtime) pairs recorded = the last events that fired
  PyObject *tail = Py_None;
  Py_INCREF(tail);
  if (c.event_timing_count != nullptr && dbg_copy(c.event_timing_count, sizeof(int), s, timeout_s)) {
    int cnt = *(int *)g_dbg_host;
    if (cnt > c.event_timing_max_entries) {
      cnt = c.event_timing_max_entries;
    }
    int n = cnt < 32 ? cnt : 32;
    if (n > 0) {
      Py_DECREF(tail);
      tail = dbg_list(c.event_timing_buffer + (size_t)(cnt - n) * 2, 2 * n, 8, s, timeout_s, &fails);
    }
  }
  PyDict_SetItemString(d, "event_timing_tail", tail);
  Py_DECREF(tail);
  DBG_INT("copy_failures", fails);
#undef DBG_INT
#undef DBG_ARR
  return d;
}

// ---- ModuleMethods entries ----
  {"launch_async_func", launch_async_func, METH_VARARGS, "launch persistent kernel on a detached thread"},
  {"launch_poll_func", launch_poll_func, METH_VARARGS, "-1.0 while the launch runs, else wall ms"},
  {"set_max_seq_length_func", set_max_seq_length_func, METH_VARARGS, "runtime termination bound"},
  {"reset_event_timing_func", reset_event_timing_func, METH_VARARGS, "event_timing_count = 0"},
  {"debug_state_func", debug_state_func, METH_VARARGS, "host-visible runtime state (watchdog)"},
