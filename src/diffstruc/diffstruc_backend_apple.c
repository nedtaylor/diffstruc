#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DIFFSTRUC_MPS_DATA_TYPE_FLOAT32 0x10000020u
#define DIFFSTRUC_CBLAS_COL_MAJOR 102
#define DIFFSTRUC_CBLAS_NO_TRANS 111
#define DIFFSTRUC_CBLAS_TRANS 112
#define DIFFSTRUC_COMMAND_BUFFER_COMPLETED 4ul
#define DIFFSTRUC_METAL_KERNEL_CACHE_SIZE 16u
#define DIFFSTRUC_METAL_BUFFER_CACHE_SIZE 12u

enum {
    DIFFSTRUC_BACKEND_AUTO = 0,
    DIFFSTRUC_BACKEND_LEGACY = 1,
    DIFFSTRUC_BACKEND_ACCELERATE = 2,
    DIFFSTRUC_BACKEND_METAL = 3
};

enum {
    DIFFSTRUC_STATUS_SUCCESS = 0,
    DIFFSTRUC_STATUS_UNAVAILABLE = 1,
    DIFFSTRUC_STATUS_ERROR = 2
};

typedef void *id;
typedef void *SEL;
typedef signed char diffstruc_objc_bool;

typedef id (*objc_getClass_t)(const char *);
typedef SEL (*sel_registerName_t)(const char *);
typedef void *(*objc_msgSend_t)(id, SEL, ...);
typedef id (*MTLCreateSystemDefaultDevice_t)(void);
typedef void (*cblas_sgemm_t)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
typedef void (*cblas_sgemv_t)(int, int, int, int, float, const float *, int, const float *, int, float, float *, int);

struct diffstruc_objc_state {
    int initialized;
    int available;
    void *handle;
    objc_getClass_t get_class;
    sel_registerName_t sel_register_name;
    objc_msgSend_t msg_send;
};

struct diffstruc_selectors {
    SEL alloc;
    SEL init;
    SEL release;
    SEL retain;
    SEL matrix_descriptor_with_rows_columns_row_bytes_data_type;
    SEL new_command_queue;
    SEL command_buffer;
    SEL new_buffer_with_bytes_no_copy;
    SEL new_buffer_with_length;
    SEL contents;
    SEL status;
    SEL commit;
    SEL wait_until_completed;
    SEL init_with_buffer_descriptor;
    SEL init_with_device_gemm;
    SEL encode_matrix_gemm;
};

struct diffstruc_accelerate_state {
    int initialized;
    int available;
    void *handle;
    cblas_sgemm_t sgemm;
    cblas_sgemv_t sgemv;
};

struct diffstruc_metal_kernel_entry {
    int used;
    unsigned char transpose_left;
    unsigned char transpose_right;
    int result_rows;
    int result_columns;
    int interior_columns;
    float alpha;
    float beta;
    id kernel;
};

struct diffstruc_metal_buffer_cache_entry {
    id buffer;
    void *mapped_ptr;
    size_t bytes;
    int in_use;
};

struct diffstruc_metal_state {
    int initialized;
    int available;
    void *metal_handle;
    void *mps_core_handle;
    void *mps_matrix_handle;
    MTLCreateSystemDefaultDevice_t create_device;
    id device;
    id queue;
    id matrix_descriptor_class;
    id matrix_class;
    id matrix_multiplication_class;
    struct diffstruc_metal_kernel_entry kernels[DIFFSTRUC_METAL_KERNEL_CACHE_SIZE];
    struct diffstruc_metal_buffer_cache_entry buffer_cache[DIFFSTRUC_METAL_BUFFER_CACHE_SIZE];
    size_t next_kernel_slot;
    long long metal_min_ops;
};

struct diffstruc_metal_buffer {
    id buffer;
    void *mapped_ptr;
    size_t bytes;
    int copy_back;
    int cache_slot;
};

static struct diffstruc_objc_state g_objc = {0};
static struct diffstruc_selectors g_sel = {0};
static struct diffstruc_accelerate_state g_accelerate = {0};
static struct diffstruc_metal_state g_metal = {0};

static int diffstruc_debug_enabled(void) {
    const char *value = getenv("DIFFSTRUC_BACKEND_DEBUG");
    return value != NULL && *value != '\0' && *value != '0';
}

static void diffstruc_debug_log(const char *message) {
    if (!diffstruc_debug_enabled()) {
        return;
    }
    fprintf(stderr, "diffstruc_backend: %s\n", message);
    fflush(stderr);
}

#define MSG_ID(obj, selector, ...) (((id (*)(id, SEL, ...)) g_objc.msg_send)((obj), (selector), __VA_ARGS__))
#define MSG_ID0(obj, selector) (((id (*)(id, SEL)) g_objc.msg_send)((obj), (selector)))
#define MSG_VOID(obj, selector, ...) (((void (*)(id, SEL, ...)) g_objc.msg_send)((obj), (selector), __VA_ARGS__))
#define MSG_VOID0(obj, selector) (((void (*)(id, SEL)) g_objc.msg_send)((obj), (selector)))
#define MSG_PTR0(obj, selector) (((void *(*)(id, SEL)) g_objc.msg_send)((obj), (selector)))
#define MSG_ULONG0(obj, selector) (((unsigned long (*)(id, SEL)) g_objc.msg_send)((obj), (selector)))

static id diffstruc_msg_id_buffer_no_copy(id object, SEL selector,
                                          void *host_ptr, unsigned long bytes,
                                          unsigned long options, void *deallocator) {
    typedef id (*msg_t)(id, SEL, void *, unsigned long, unsigned long, void *);
    return ((msg_t) g_objc.msg_send)(object, selector, host_ptr, bytes, options, deallocator);
}

static id diffstruc_msg_id_length_options(id object, SEL selector,
                                          unsigned long bytes, unsigned long options) {
    typedef id (*msg_t)(id, SEL, unsigned long, unsigned long);
    return ((msg_t) g_objc.msg_send)(object, selector, bytes, options);
}

static id diffstruc_msg_id_descriptor(id object, SEL selector,
                                      unsigned long rows, unsigned long columns,
                                      unsigned long row_bytes, unsigned int data_type) {
    typedef id (*msg_t)(id, SEL, unsigned long, unsigned long, unsigned long, unsigned int);
    return ((msg_t) g_objc.msg_send)(object, selector, rows, columns, row_bytes, data_type);
}

static id diffstruc_msg_id_id_id(id object, SEL selector, id arg1, id arg2) {
    typedef id (*msg_t)(id, SEL, id, id);
    return ((msg_t) g_objc.msg_send)(object, selector, arg1, arg2);
}

static id diffstruc_msg_id_gemm(id object, SEL selector, id device,
                                diffstruc_objc_bool transpose_left,
                                diffstruc_objc_bool transpose_right,
                                unsigned long result_rows,
                                unsigned long result_columns,
                                unsigned long interior_columns,
                                double alpha, double beta) {
    typedef id (*msg_t)(id, SEL, id, diffstruc_objc_bool, diffstruc_objc_bool,
                        unsigned long, unsigned long, unsigned long, double, double);
    return ((msg_t) g_objc.msg_send)(object, selector, device, transpose_left,
                                     transpose_right, result_rows, result_columns,
                                     interior_columns, alpha, beta);
}

static void diffstruc_msg_void_encode_gemm(id object, SEL selector,
                                           id command_buffer, id left_matrix,
                                           id right_matrix, id result_matrix) {
    typedef void (*msg_t)(id, SEL, id, id, id, id);
    ((msg_t) g_objc.msg_send)(object, selector, command_buffer, left_matrix,
                              right_matrix, result_matrix);
}

static char diffstruc_uppercase(char value) {
    if (value >= 'a' && value <= 'z') {
        return (char) (value - ('a' - 'A'));
    }
    return value;
}

static int diffstruc_load_objc(void) {
#if !defined(__APPLE__)
    return 0;
#else
    const char *objc_paths[] = {
        "/usr/lib/libobjc.A.dylib",
        "/usr/lib/libobjc.dylib"
    };
    size_t i;

    if (g_objc.initialized) {
        return g_objc.available;
    }

    g_objc.initialized = 1;
    for (i = 0; i < sizeof(objc_paths) / sizeof(objc_paths[0]); ++i) {
        g_objc.handle = dlopen(objc_paths[i], RTLD_LAZY | RTLD_LOCAL);
        if (g_objc.handle != NULL) {
            break;
        }
    }
    if (g_objc.handle == NULL) {
        return 0;
    }

    g_objc.get_class = (objc_getClass_t) dlsym(g_objc.handle, "objc_getClass");
    g_objc.sel_register_name = (sel_registerName_t) dlsym(g_objc.handle, "sel_registerName");
    g_objc.msg_send = (objc_msgSend_t) dlsym(g_objc.handle, "objc_msgSend");
    if (g_objc.get_class == NULL || g_objc.sel_register_name == NULL || g_objc.msg_send == NULL) {
        return 0;
    }

    g_sel.alloc = g_objc.sel_register_name("alloc");
    g_sel.init = g_objc.sel_register_name("init");
    g_sel.release = g_objc.sel_register_name("release");
    g_sel.retain = g_objc.sel_register_name("retain");
    g_sel.matrix_descriptor_with_rows_columns_row_bytes_data_type =
        g_objc.sel_register_name("matrixDescriptorWithRows:columns:rowBytes:dataType:");
    g_sel.new_command_queue = g_objc.sel_register_name("newCommandQueue");
    g_sel.command_buffer = g_objc.sel_register_name("commandBuffer");
    g_sel.new_buffer_with_bytes_no_copy = g_objc.sel_register_name("newBufferWithBytesNoCopy:length:options:deallocator:");
    g_sel.new_buffer_with_length = g_objc.sel_register_name("newBufferWithLength:options:");
    g_sel.contents = g_objc.sel_register_name("contents");
    g_sel.status = g_objc.sel_register_name("status");
    g_sel.commit = g_objc.sel_register_name("commit");
    g_sel.wait_until_completed = g_objc.sel_register_name("waitUntilCompleted");
    g_sel.init_with_buffer_descriptor = g_objc.sel_register_name("initWithBuffer:descriptor:");
    g_sel.init_with_device_gemm = g_objc.sel_register_name("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:");
    g_sel.encode_matrix_gemm = g_objc.sel_register_name("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:");

    g_objc.available = 1;
    return 1;
#endif
}

static int diffstruc_load_accelerate(void) {
#if !defined(__APPLE__)
    return 0;
#else
    if (g_accelerate.initialized) {
        return g_accelerate.available;
    }

    g_accelerate.initialized = 1;
    g_accelerate.handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_LAZY | RTLD_LOCAL);
    if (g_accelerate.handle == NULL) {
        return 0;
    }

    g_accelerate.sgemm = (cblas_sgemm_t) dlsym(g_accelerate.handle, "cblas_sgemm");
    g_accelerate.sgemv = (cblas_sgemv_t) dlsym(g_accelerate.handle, "cblas_sgemv");
    if (g_accelerate.sgemm == NULL || g_accelerate.sgemv == NULL) {
        return 0;
    }

    g_accelerate.available = 1;
    return 1;
#endif
}

static long long diffstruc_default_metal_threshold(void) {
    const char *env_value;
    char *end_ptr;
    long long parsed_value;
    const long long default_threshold = 268435456LL;

    env_value = getenv("DIFFSTRUC_METAL_MIN_OPS");
    if (env_value == NULL || *env_value == '\0') {
        return default_threshold;
    }

    parsed_value = strtoll(env_value, &end_ptr, 10);
    if (end_ptr == env_value || parsed_value <= 0) {
        return default_threshold;
    }
    return parsed_value;
}

static int diffstruc_load_metal(void) {
#if !defined(__APPLE__)
    return 0;
#else
    if (g_metal.initialized) {
        diffstruc_debug_log(g_metal.available ? "metal already initialised and available" : "metal already initialised and unavailable");
        return g_metal.available;
    }

    g_metal.initialized = 1;
    diffstruc_debug_log("starting metal initialisation");
    if (!diffstruc_load_objc()) {
        diffstruc_debug_log("objc runtime unavailable");
        return 0;
    }

    diffstruc_debug_log("opening Metal frameworks");
    g_metal.metal_handle = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY | RTLD_GLOBAL);
    g_metal.mps_core_handle = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/Frameworks/MPSCore.framework/MPSCore", RTLD_LAZY | RTLD_GLOBAL);
    g_metal.mps_matrix_handle = dlopen("/System/Library/Frameworks/MetalPerformanceShaders.framework/Frameworks/MPSMatrix.framework/MPSMatrix", RTLD_LAZY | RTLD_GLOBAL);
    if (g_metal.metal_handle == NULL || g_metal.mps_core_handle == NULL || g_metal.mps_matrix_handle == NULL) {
        diffstruc_debug_log("failed to open Metal/MPS frameworks");
        return 0;
    }

    g_metal.create_device = (MTLCreateSystemDefaultDevice_t) dlsym(g_metal.metal_handle, "MTLCreateSystemDefaultDevice");
    if (g_metal.create_device == NULL) {
        diffstruc_debug_log("MTLCreateSystemDefaultDevice not found");
        return 0;
    }

    diffstruc_debug_log("creating default Metal device");
    g_metal.device = g_metal.create_device();
    if (g_metal.device == NULL) {
        diffstruc_debug_log("no default Metal device available");
        return 0;
    }

    diffstruc_debug_log("creating command queue");
    g_metal.queue = MSG_ID0(g_metal.device, g_sel.new_command_queue);
    if (g_metal.queue == NULL) {
        diffstruc_debug_log("failed to create Metal command queue");
        return 0;
    }

    diffstruc_debug_log("resolving MPS classes");
    g_metal.matrix_descriptor_class = g_objc.get_class("MPSMatrixDescriptor");
    g_metal.matrix_class = g_objc.get_class("MPSMatrix");
    g_metal.matrix_multiplication_class = g_objc.get_class("MPSMatrixMultiplication");
    if (g_metal.matrix_descriptor_class == NULL || g_metal.matrix_class == NULL || g_metal.matrix_multiplication_class == NULL) {
        diffstruc_debug_log("required MPS classes unavailable");
        return 0;
    }

    g_metal.metal_min_ops = diffstruc_default_metal_threshold();
    g_metal.available = 1;
    diffstruc_debug_log("metal initialisation complete");
    return 1;
#endif
}

static int diffstruc_should_use_metal(int m, int n, int k) {
    long long ops;

    if (m < 64 || n < 64 || k < 64) {
        return 0;
    }
    ops = (long long) m * (long long) n * (long long) k;
    return ops >= g_metal.metal_min_ops;
}

static int diffstruc_cblas_transpose(char trans) {
    return diffstruc_uppercase(trans) == 'T' ? DIFFSTRUC_CBLAS_TRANS : DIFFSTRUC_CBLAS_NO_TRANS;
}

static int diffstruc_accelerate_sgemm(char transa, char transb, int m, int n, int k,
                                      float alpha, const float *a, int lda,
                                      const float *b, int ldb,
                                      float beta, float *c, int ldc) {
    if (!diffstruc_load_accelerate()) {
        return DIFFSTRUC_STATUS_UNAVAILABLE;
    }

    g_accelerate.sgemm(DIFFSTRUC_CBLAS_COL_MAJOR,
                       diffstruc_cblas_transpose(transa),
                       diffstruc_cblas_transpose(transb),
                       m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    return DIFFSTRUC_STATUS_SUCCESS;
}

static int diffstruc_accelerate_sgemv(char trans, int m, int n, float alpha,
                                      const float *a, int lda, const float *x, int incx,
                                      float beta, float *y, int incy) {
    if (!diffstruc_load_accelerate()) {
        return DIFFSTRUC_STATUS_UNAVAILABLE;
    }

    g_accelerate.sgemv(DIFFSTRUC_CBLAS_COL_MAJOR,
                       diffstruc_cblas_transpose(trans),
                       m, n, alpha, a, lda, x, incx, beta, y, incy);
    return DIFFSTRUC_STATUS_SUCCESS;
}

static id diffstruc_create_matrix_descriptor(unsigned long rows, unsigned long columns, unsigned long row_bytes) {
    id descriptor;

    descriptor = diffstruc_msg_id_descriptor(
        g_metal.matrix_descriptor_class,
        g_sel.matrix_descriptor_with_rows_columns_row_bytes_data_type,
        rows,
        columns,
        row_bytes,
        (unsigned int) DIFFSTRUC_MPS_DATA_TYPE_FLOAT32);
    if (descriptor == NULL) {
        return NULL;
    }

    descriptor = MSG_ID0(descriptor, g_sel.retain);
    if (descriptor == NULL) {
        return NULL;
    }
    return descriptor;
}

static id diffstruc_create_matrix(id buffer, id descriptor) {
    id matrix;

    matrix = MSG_ID0(g_metal.matrix_class, g_sel.alloc);
    if (matrix == NULL) {
        return NULL;
    }
    matrix = diffstruc_msg_id_id_id(matrix, g_sel.init_with_buffer_descriptor, buffer, descriptor);
    return matrix;
}

static id diffstruc_get_cached_kernel(unsigned char transpose_left, unsigned char transpose_right,
                                      int result_rows, int result_columns, int interior_columns,
                                      float alpha, float beta) {
    size_t index;
    struct diffstruc_metal_kernel_entry *entry;
    id kernel;

    for (index = 0; index < sizeof(g_metal.kernels) / sizeof(g_metal.kernels[0]); ++index) {
        entry = &g_metal.kernels[index];
        if (!entry->used) {
            continue;
        }
        if (entry->transpose_left == transpose_left &&
            entry->transpose_right == transpose_right &&
            entry->result_rows == result_rows &&
            entry->result_columns == result_columns &&
            entry->interior_columns == interior_columns &&
            entry->alpha == alpha &&
            entry->beta == beta) {
            return entry->kernel;
        }
    }

    kernel = MSG_ID0(g_metal.matrix_multiplication_class, g_sel.alloc);
    if (kernel == NULL) {
        return NULL;
    }
    kernel = diffstruc_msg_id_gemm(kernel, g_sel.init_with_device_gemm,
                                   g_metal.device,
                                   (diffstruc_objc_bool) (transpose_left ? 1 : 0),
                                   (diffstruc_objc_bool) (transpose_right ? 1 : 0),
                                   (unsigned long) result_rows,
                                   (unsigned long) result_columns,
                                   (unsigned long) interior_columns,
                                   (double) alpha,
                                   (double) beta);
    if (kernel == NULL) {
        return NULL;
    }

    entry = &g_metal.kernels[g_metal.next_kernel_slot % DIFFSTRUC_METAL_KERNEL_CACHE_SIZE];
    if (entry->used && entry->kernel != NULL) {
        MSG_VOID0(entry->kernel, g_sel.release);
    }
    entry->used = 1;
    entry->transpose_left = transpose_left;
    entry->transpose_right = transpose_right;
    entry->result_rows = result_rows;
    entry->result_columns = result_columns;
    entry->interior_columns = interior_columns;
    entry->alpha = alpha;
    entry->beta = beta;
    entry->kernel = kernel;
    g_metal.next_kernel_slot += 1;
    return kernel;
}

static int diffstruc_allocate_shared_buffer(size_t bytes, id *buffer_out, void **mapped_ptr_out) {
    id buffer;
    void *mapped_ptr;

    buffer = diffstruc_msg_id_length_options(g_metal.device, g_sel.new_buffer_with_length,
                                             (unsigned long) bytes, (unsigned long) 0);
    if (buffer == NULL) {
        return 0;
    }

    mapped_ptr = MSG_PTR0(buffer, g_sel.contents);
    if (mapped_ptr == NULL) {
        MSG_VOID0(buffer, g_sel.release);
        return 0;
    }

    *buffer_out = buffer;
    *mapped_ptr_out = mapped_ptr;
    return 1;
}

static int diffstruc_acquire_cached_buffer(size_t bytes,
                                           struct diffstruc_metal_buffer *out_buffer) {
    size_t index;
    long candidate_index = -1;
    long grow_index = -1;
    long empty_index = -1;
    size_t candidate_bytes = 0;
    struct diffstruc_metal_buffer_cache_entry *entry;

    for (index = 0; index < DIFFSTRUC_METAL_BUFFER_CACHE_SIZE; ++index) {
        entry = &g_metal.buffer_cache[index];
        if (entry->buffer == NULL) {
            if (empty_index < 0) {
                empty_index = (long) index;
            }
            continue;
        }
        if (entry->in_use) {
            continue;
        }
        if (entry->bytes >= bytes) {
            if (candidate_index < 0 || entry->bytes < candidate_bytes) {
                candidate_index = (long) index;
                candidate_bytes = entry->bytes;
            }
            continue;
        }
        if (grow_index < 0 || entry->bytes < g_metal.buffer_cache[grow_index].bytes) {
            grow_index = (long) index;
        }
    }

    if (candidate_index < 0) {
        candidate_index = empty_index >= 0 ? empty_index : grow_index;
        if (candidate_index < 0) {
            return 0;
        }

        entry = &g_metal.buffer_cache[candidate_index];
        if (entry->buffer != NULL) {
            MSG_VOID0(entry->buffer, g_sel.release);
            entry->buffer = NULL;
            entry->mapped_ptr = NULL;
            entry->bytes = 0;
        }

        if (!diffstruc_allocate_shared_buffer(bytes, &entry->buffer, &entry->mapped_ptr)) {
            return 0;
        }
        entry->bytes = bytes;
    } else {
        entry = &g_metal.buffer_cache[candidate_index];
    }

    entry->in_use = 1;
    out_buffer->buffer = entry->buffer;
    out_buffer->mapped_ptr = entry->mapped_ptr;
    out_buffer->bytes = bytes;
    out_buffer->copy_back = 0;
    out_buffer->cache_slot = (int) candidate_index;
    return 1;
}

static int diffstruc_create_metal_buffer(const float *host_ptr, size_t bytes,
                                         int initialise_from_host, int copy_back,
                                         struct diffstruc_metal_buffer *out_buffer) {
    id buffer;

    memset(out_buffer, 0, sizeof(*out_buffer));
    out_buffer->cache_slot = -1;
    buffer = diffstruc_msg_id_buffer_no_copy(g_metal.device, g_sel.new_buffer_with_bytes_no_copy,
                                             (void *) host_ptr, (unsigned long) bytes,
                                             (unsigned long) 0, (void *) 0);
    if (buffer != NULL) {
        out_buffer->buffer = buffer;
        out_buffer->bytes = bytes;
        out_buffer->copy_back = 0;
        return 1;
    }

    if (!diffstruc_acquire_cached_buffer(bytes, out_buffer)) {
        return 0;
    }

    if (initialise_from_host) {
        memcpy(out_buffer->mapped_ptr, host_ptr, bytes);
    }

    out_buffer->copy_back = copy_back;
    return 1;
}

static void diffstruc_release_metal_buffer(struct diffstruc_metal_buffer *buffer, float *host_ptr) {
    if (buffer->copy_back && buffer->mapped_ptr != NULL && host_ptr != NULL) {
        memcpy(host_ptr, buffer->mapped_ptr, buffer->bytes);
    }
    if (buffer->cache_slot >= 0) {
        g_metal.buffer_cache[buffer->cache_slot].in_use = 0;
        return;
    }
    if (buffer->buffer != NULL) {
        MSG_VOID0(buffer->buffer, g_sel.release);
    }
}

static int diffstruc_metal_sgemm(char transa, char transb, int m, int n, int k,
                                 float alpha, const float *a, int lda,
                                 const float *b, int ldb,
                                 float beta, float *c, int ldc) {
    unsigned char transpose_left;
    unsigned char transpose_right;
    unsigned long rows_a;
    unsigned long cols_a;
    unsigned long rows_b;
    unsigned long cols_b;
    size_t bytes_a;
    size_t bytes_b;
    size_t bytes_c;
    id kernel;
    id desc_a = NULL;
    id desc_b = NULL;
    id desc_c = NULL;
    id mat_a = NULL;
    id mat_b = NULL;
    id mat_c = NULL;
    id command_buffer = NULL;
    unsigned long status;
    struct diffstruc_metal_buffer buffer_a;
    struct diffstruc_metal_buffer buffer_b;
    struct diffstruc_metal_buffer buffer_c;

    if (!diffstruc_load_metal()) {
        return DIFFSTRUC_STATUS_UNAVAILABLE;
    }

    transpose_left = (unsigned char) (diffstruc_uppercase(transb) == 'T');
    transpose_right = (unsigned char) (diffstruc_uppercase(transa) == 'T');
    kernel = diffstruc_get_cached_kernel(transpose_left, transpose_right, n, m, k, alpha, beta);
    if (kernel == NULL) {
        return DIFFSTRUC_STATUS_ERROR;
    }

    rows_a = (unsigned long) ((diffstruc_uppercase(transa) == 'T') ? m : k);
    cols_a = (unsigned long) ((diffstruc_uppercase(transa) == 'T') ? k : m);
    rows_b = (unsigned long) ((diffstruc_uppercase(transb) == 'T') ? k : n);
    cols_b = (unsigned long) ((diffstruc_uppercase(transb) == 'T') ? n : k);
    bytes_a = (size_t) lda * rows_a * sizeof(float);
    bytes_b = (size_t) ldb * rows_b * sizeof(float);
    bytes_c = (size_t) ldc * (size_t) n * sizeof(float);

    if (!diffstruc_create_metal_buffer(a, bytes_a, 0, 0, &buffer_a)) {
        return DIFFSTRUC_STATUS_ERROR;
    }
    if (!diffstruc_create_metal_buffer(b, bytes_b, 0, 0, &buffer_b)) {
        diffstruc_release_metal_buffer(&buffer_a, NULL);
        return DIFFSTRUC_STATUS_ERROR;
    }
    if (!diffstruc_create_metal_buffer(c, bytes_c, beta != 0.0f, 1, &buffer_c)) {
        diffstruc_release_metal_buffer(&buffer_b, NULL);
        diffstruc_release_metal_buffer(&buffer_a, NULL);
        return DIFFSTRUC_STATUS_ERROR;
    }

    desc_a = diffstruc_create_matrix_descriptor(rows_a, cols_a, (unsigned long) lda * sizeof(float));
    desc_b = diffstruc_create_matrix_descriptor(rows_b, cols_b, (unsigned long) ldb * sizeof(float));
    desc_c = diffstruc_create_matrix_descriptor((unsigned long) n, (unsigned long) m, (unsigned long) ldc * sizeof(float));
    if (desc_a == NULL || desc_b == NULL || desc_c == NULL) {
        goto metal_cleanup;
    }

    mat_a = diffstruc_create_matrix(buffer_a.buffer, desc_a);
    mat_b = diffstruc_create_matrix(buffer_b.buffer, desc_b);
    mat_c = diffstruc_create_matrix(buffer_c.buffer, desc_c);
    if (mat_a == NULL || mat_b == NULL || mat_c == NULL) {
        goto metal_cleanup;
    }

    command_buffer = MSG_ID0(g_metal.queue, g_sel.command_buffer);
    if (command_buffer == NULL) {
        goto metal_cleanup;
    }
    command_buffer = MSG_ID0(command_buffer, g_sel.retain);
    if (command_buffer == NULL) {
        goto metal_cleanup;
    }

    diffstruc_msg_void_encode_gemm(kernel, g_sel.encode_matrix_gemm,
                                   command_buffer, mat_b, mat_a, mat_c);
    MSG_VOID0(command_buffer, g_sel.commit);
    MSG_VOID0(command_buffer, g_sel.wait_until_completed);
    status = MSG_ULONG0(command_buffer, g_sel.status);
    if (status != DIFFSTRUC_COMMAND_BUFFER_COMPLETED) {
        goto metal_cleanup;
    }

    if (buffer_c.copy_back && buffer_c.mapped_ptr != NULL) {
        memcpy(c, buffer_c.mapped_ptr, buffer_c.bytes);
    }

    if (command_buffer != NULL) {
        MSG_VOID0(command_buffer, g_sel.release);
    }
    if (mat_c != NULL) {
        MSG_VOID0(mat_c, g_sel.release);
    }
    if (mat_b != NULL) {
        MSG_VOID0(mat_b, g_sel.release);
    }
    if (mat_a != NULL) {
        MSG_VOID0(mat_a, g_sel.release);
    }
    if (desc_c != NULL) {
        MSG_VOID0(desc_c, g_sel.release);
    }
    if (desc_b != NULL) {
        MSG_VOID0(desc_b, g_sel.release);
    }
    if (desc_a != NULL) {
        MSG_VOID0(desc_a, g_sel.release);
    }
    diffstruc_release_metal_buffer(&buffer_c, NULL);
    diffstruc_release_metal_buffer(&buffer_b, NULL);
    diffstruc_release_metal_buffer(&buffer_a, NULL);
    return DIFFSTRUC_STATUS_SUCCESS;

metal_cleanup:
    if (command_buffer != NULL) {
        MSG_VOID0(command_buffer, g_sel.release);
    }
    if (mat_c != NULL) {
        MSG_VOID0(mat_c, g_sel.release);
    }
    if (mat_b != NULL) {
        MSG_VOID0(mat_b, g_sel.release);
    }
    if (mat_a != NULL) {
        MSG_VOID0(mat_a, g_sel.release);
    }
    if (desc_c != NULL) {
        MSG_VOID0(desc_c, g_sel.release);
    }
    if (desc_b != NULL) {
        MSG_VOID0(desc_b, g_sel.release);
    }
    if (desc_a != NULL) {
        MSG_VOID0(desc_a, g_sel.release);
    }
    diffstruc_release_metal_buffer(&buffer_c, c);
    diffstruc_release_metal_buffer(&buffer_b, NULL);
    diffstruc_release_metal_buffer(&buffer_a, NULL);
    return DIFFSTRUC_STATUS_ERROR;
}

int diffstruc_apple_backend_available(int backend) {
    if (diffstruc_debug_enabled()) {
        fprintf(stderr, "diffstruc_backend: probing backend %d\n", backend);
        fflush(stderr);
    }
    switch (backend) {
        case DIFFSTRUC_BACKEND_ACCELERATE:
            return diffstruc_load_accelerate() ? 1 : 0;
        case DIFFSTRUC_BACKEND_METAL:
            return diffstruc_load_metal() ? 1 : 0;
        default:
            return 0;
    }
}

int diffstruc_apple_resolve_backend(int backend, int m, int n, int k, int is_gemv) {
    if (backend == DIFFSTRUC_BACKEND_METAL) {
        if (!is_gemv && diffstruc_load_metal()) {
            return DIFFSTRUC_BACKEND_METAL;
        }
        if (diffstruc_load_accelerate()) {
            return DIFFSTRUC_BACKEND_ACCELERATE;
        }
        return 0;
    }

    if (backend == DIFFSTRUC_BACKEND_ACCELERATE) {
        return diffstruc_load_accelerate() ? DIFFSTRUC_BACKEND_ACCELERATE : 0;
    }

    if (!is_gemv && diffstruc_load_metal() && diffstruc_should_use_metal(m, n, k)) {
        return DIFFSTRUC_BACKEND_METAL;
    }
    if (diffstruc_load_accelerate()) {
        return DIFFSTRUC_BACKEND_ACCELERATE;
    }
    return 0;
}

int diffstruc_apple_sgemm(int backend, char transa, char transb, int m, int n, int k,
                          float alpha, const float *a, int lda,
                          const float *b, int ldb,
                          float beta, float *c, int ldc) {
    int resolved_backend;

    resolved_backend = backend;
    if (backend == DIFFSTRUC_BACKEND_AUTO) {
        resolved_backend = diffstruc_apple_resolve_backend(backend, m, n, k, 0);
    }

    if (resolved_backend == DIFFSTRUC_BACKEND_METAL) {
        int status = diffstruc_metal_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        if (status == DIFFSTRUC_STATUS_SUCCESS) {
            return status;
        }
        if (backend == DIFFSTRUC_BACKEND_METAL) {
            return diffstruc_accelerate_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }
    }

    return diffstruc_accelerate_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

int diffstruc_apple_sgemv(int backend, char trans, int m, int n, float alpha,
                          const float *a, int lda, const float *x, int incx,
                          float beta, float *y, int incy) {
    int resolved_backend;

    resolved_backend = backend;
    if (backend == DIFFSTRUC_BACKEND_AUTO) {
        resolved_backend = diffstruc_apple_resolve_backend(backend, m, n, 1, 1);
    }
    if (resolved_backend == DIFFSTRUC_BACKEND_METAL) {
        resolved_backend = DIFFSTRUC_BACKEND_ACCELERATE;
    }
    if (resolved_backend != DIFFSTRUC_BACKEND_ACCELERATE) {
        return DIFFSTRUC_STATUS_UNAVAILABLE;
    }
    return diffstruc_accelerate_sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
