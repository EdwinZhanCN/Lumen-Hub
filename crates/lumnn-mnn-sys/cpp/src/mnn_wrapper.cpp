// Derived from rust-paddle-ocr
// Original project: https://github.com/zibo-chen/rust-paddle-ocr
// Licensed under the Apache License, Version 2.0
// Modifications made for this project.
#include "mnn_wrapper.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <MNN/MNNDefine.h>

#include <cstring>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <memory>
#include <map>

// C++11 compatible make_unique
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_ptr(Args &&...args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Global mutex to serialize MNN inference calls
// MNN's internal thread pool has a limit of MNN_THREAD_POOL_MAX_TASKS (default=2)
// This mutex ensures only one inference runs at a time, avoiding thread pool exhaustion
static std::mutex g_mnn_inference_mutex;

// ============== Internal Structures ==============

struct MNN_SharedRuntime
{
    MNN::BackendConfig backend_config;
    MNN::ScheduleConfig schedule_config;
    int thread_count;
    int precision_mode;
};

struct MNN_InferenceEngine
{
    std::unique_ptr<MNN::Interpreter> interpreter;
    MNN::Session *default_session;
    std::mutex mutex;
    std::string last_error;

    // All I/O tensors (name → tensor)
    std::map<std::string, MNN::Tensor*> input_tensors;
    std::map<std::string, MNN::Tensor*> output_tensors;
    // Ordered names for index-based access
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    MNN_SharedRuntime *runtime; // Optional shared runtime
    bool owns_runtime;
    int forward_type; // Backend type set at creation time

    MNN_InferenceEngine() : default_session(nullptr),
                            runtime(nullptr), owns_runtime(false),
                            forward_type(MNN_FORWARD_CPU) {}
};

struct MNN_SingleSession
{
    MNN::Session *session;
    MNN_InferenceEngine *engine;
    std::string last_error;
    MNN::Tensor *input_tensor;
    MNN::Tensor *output_tensor;

    MNN_SingleSession() : session(nullptr), engine(nullptr),
                          input_tensor(nullptr), output_tensor(nullptr) {}
};

struct MNN_SessionPool
{
    MNN_InferenceEngine *engine;
    std::vector<MNN::Session *> sessions;
    std::vector<MNN::Tensor *> input_tensors;
    std::vector<MNN::Tensor *> output_tensors;

    std::mutex mutex;
    std::condition_variable cv;
    std::queue<size_t> available_sessions;
    std::string last_error;
};

// ============== Helper Functions ==============

// Initialize schedule and backend configs from MNNR_Config.
// Caller must ensure `schedule` and `backend` outlive any use of schedule.backendConfig.
static void init_schedule_config(MNN::ScheduleConfig &schedule, MNN::BackendConfig &backend, const MNNR_Config *config)
{
    schedule.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    schedule.numThread = config ? config->thread_count : 4;
    if (schedule.numThread <= 0)
    {
        schedule.numThread = 4;
    }

    if (config)
    {
        switch (config->precision_mode)
        {
        case 1:
            backend.precision = MNN::BackendConfig::Precision_Low;
            break;
        case 2:
            backend.precision = MNN::BackendConfig::Precision_High;
            break;
        default:
            backend.precision = MNN::BackendConfig::Precision_Normal;
            break;
        }
    }
    schedule.backendConfig = &backend;
}

static bool init_engine_tensors(MNN_InferenceEngine *engine)
{
    if (!engine->interpreter || !engine->default_session)
    {
        return false;
    }

    // Get all input tensors
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found";
        return false;
    }
    for (auto &kv : input_map)
    {
        engine->input_tensors[kv.first] = kv.second;
        engine->input_names.push_back(kv.first);
    }

    // Get all output tensors
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return false;
    }
    for (auto &kv : output_map)
    {
        engine->output_tensors[kv.first] = kv.second;
        engine->output_names.push_back(kv.first);
    }

    return true;
}

static size_t tensor_element_count(const MNN::Tensor *tensor)
{
    if (!tensor)
    {
        return 0;
    }
    size_t count = 1;
    for (int dim : tensor->shape())
    {
        if (dim <= 0)
        {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

template <typename T>
static MNNR_ErrorCode copy_input_typed(MNN_InferenceEngine *engine, const char *name, const T *data, size_t size)
{
    if (!engine || !name || !data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    std::lock_guard<std::mutex> lock(engine->mutex);
    auto it = engine->input_tensors.find(name);
    if (it == engine->input_tensors.end())
    {
        engine->last_error = std::string("Input tensor not found: ") + name;
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    size_t expected_size = tensor_element_count(it->second);
    if (expected_size != 0 && expected_size != size)
    {
        engine->last_error = std::string("Input tensor size mismatch for ") + name;
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    auto host = make_unique_ptr<MNN::Tensor>(it->second, MNN::Tensor::CAFFE);
    std::memcpy(host->host<T>(), data, size * sizeof(T));
    it->second->copyFromHostTensor(host.get());
    return MNNR_SUCCESS;
}

template <typename T>
static MNNR_ErrorCode copy_output_typed(MNN_InferenceEngine *engine, const char *name, T *data, size_t size)
{
    if (!engine || !name || !data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    std::lock_guard<std::mutex> lock(engine->mutex);
    auto it = engine->output_tensors.find(name);
    if (it == engine->output_tensors.end())
    {
        engine->last_error = std::string("Output tensor not found: ") + name;
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    size_t expected_size = tensor_element_count(it->second);
    if (expected_size != 0 && expected_size != size)
    {
        engine->last_error = std::string("Output tensor size mismatch for ") + name;
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    auto host = make_unique_ptr<MNN::Tensor>(it->second, MNN::Tensor::CAFFE);
    it->second->copyToHostTensor(host.get());
    std::memcpy(data, host->host<T>(), size * sizeof(T));
    return MNNR_SUCCESS;
}

// ============== Version & Info ==============

const char *mnnr_get_version(void)
{
    return MNN_VERSION;
}

int mnnr_get_backend_type(const MNN_InferenceEngine *engine)
{
    if (!engine)
    {
        return -1;
    }
    return engine->forward_type;
}

const char *mnnr_get_input_name(const MNN_InferenceEngine *engine)
{
    if (!engine || engine->input_names.empty())
    {
        return nullptr;
    }
    return engine->input_names[0].c_str();
}

const char *mnnr_get_output_name(const MNN_InferenceEngine *engine)
{
    if (!engine || engine->output_names.empty())
    {
        return nullptr;
    }
    return engine->output_names[0].c_str();
}

// ============== Shared Runtime API ==============

MNN_SharedRuntime *mnnr_create_runtime(const MNNR_Config *config)
{
    auto runtime = new MNN_SharedRuntime();

    runtime->thread_count = config ? config->thread_count : 4;
    if (runtime->thread_count <= 0)
    {
        runtime->thread_count = 4;
    }

    runtime->precision_mode = config ? config->precision_mode : 0;

    runtime->schedule_config.type = (config) ? static_cast<MNNForwardType>(config->forward_type) : MNN_FORWARD_CPU;
    runtime->schedule_config.numThread = runtime->thread_count;

    switch (runtime->precision_mode)
    {
    case 1:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Low;
        break;
    case 2:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_High;
        break;
    default:
        runtime->backend_config.precision = MNN::BackendConfig::Precision_Normal;
        break;
    }
    runtime->schedule_config.backendConfig = &runtime->backend_config;

    return runtime;
}

void mnnr_destroy_runtime(MNN_SharedRuntime *runtime)
{
    delete runtime;
}

// ============== Inference Engine API ==============

MNN_InferenceEngine *mnnr_create_engine(
    const void *buffer,
    size_t size,
    const MNNR_Config *config)
{
    if (!buffer || size == 0)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create default session
    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    engine->forward_type = schedule.type;
    engine->default_session = engine->interpreter->createSession(schedule);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create default session";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

MNN_InferenceEngine *mnnr_create_engine_with_runtime(
    const void *buffer,
    size_t size,
    MNN_SharedRuntime *runtime)
{
    if (!buffer || size == 0 || !runtime)
    {
        return nullptr;
    }

    auto engine = new MNN_InferenceEngine();
    engine->runtime = runtime;
    engine->owns_runtime = false;
    engine->forward_type = runtime->schedule_config.type;

    // Create interpreter from buffer
    engine->interpreter.reset(MNN::Interpreter::createFromBuffer(buffer, size));
    if (!engine->interpreter)
    {
        engine->last_error = "Failed to create interpreter from buffer";
        delete engine;
        return nullptr;
    }

    // Create session using shared runtime config
    engine->default_session = engine->interpreter->createSession(runtime->schedule_config);
    if (!engine->default_session)
    {
        engine->last_error = "Failed to create session with shared runtime";
        delete engine;
        return nullptr;
    }

    // Initialize tensors
    if (!init_engine_tensors(engine))
    {
        delete engine;
        return nullptr;
    }

    return engine;
}

void mnnr_destroy_engine(MNN_InferenceEngine *engine)
{
    if (engine)
    {
        if (engine->default_session && engine->interpreter)
        {
            engine->interpreter->releaseSession(engine->default_session);
        }
        delete engine;
    }
}

MNNR_ErrorCode mnnr_get_input_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims || engine->input_names.empty())
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    // Backward compat: return shape of first input
    return mnnr_get_input_shape_at(engine, (size_t)0, dims, out_ndims);
}

MNNR_ErrorCode mnnr_get_output_shape(
    const MNN_InferenceEngine *engine,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims || engine->output_names.empty())
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }
    // Backward compat: return shape of first output
    return mnnr_get_output_shape_at(engine, (size_t)0, dims, out_ndims);
}

// ============== Multi-I/O Query API ==============

size_t mnnr_get_input_count(const MNN_InferenceEngine *engine)
{
    if (!engine) return 0;
    return engine->input_names.size();
}

size_t mnnr_get_output_count(const MNN_InferenceEngine *engine)
{
    if (!engine) return 0;
    return engine->output_names.size();
}

const char *mnnr_get_input_name_at(const MNN_InferenceEngine *engine, size_t index)
{
    if (!engine || index >= engine->input_names.size()) return nullptr;
    return engine->input_names[index].c_str();
}

const char *mnnr_get_output_name_at(const MNN_InferenceEngine *engine, size_t index)
{
    if (!engine || index >= engine->output_names.size()) return nullptr;
    return engine->output_names[index].c_str();
}

MNNR_ErrorCode mnnr_get_input_shape_at(
    const MNN_InferenceEngine *engine,
    size_t index,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims || index >= engine->input_names.size())
        return MNNR_ERROR_INVALID_PARAMETER;

    const auto &name = engine->input_names[index];
    auto it = engine->input_tensors.find(name);
    if (it == engine->input_tensors.end()) return MNNR_ERROR_INVALID_PARAMETER;

    auto shape = it->second->shape();
    size_t ndims = shape.size() < 8 ? shape.size() : 8;
    *out_ndims = ndims;
    for (size_t i = 0; i < ndims; i++)
        dims[i] = static_cast<size_t>(shape[i]);
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_output_shape_at(
    const MNN_InferenceEngine *engine,
    size_t index,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !dims || !out_ndims || index >= engine->output_names.size())
        return MNNR_ERROR_INVALID_PARAMETER;

    const auto &name = engine->output_names[index];
    auto it = engine->output_tensors.find(name);
    if (it == engine->output_tensors.end()) return MNNR_ERROR_INVALID_PARAMETER;

    auto shape = it->second->shape();
    size_t ndims = shape.size() < 8 ? shape.size() : 8;
    *out_ndims = ndims;
    for (size_t i = 0; i < ndims; i++)
        dims[i] = static_cast<size_t>(shape[i]);
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_input_shape_by_name(
    const MNN_InferenceEngine *engine,
    const char *name,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !name || !dims || !out_ndims) return MNNR_ERROR_INVALID_PARAMETER;
    auto it = engine->input_tensors.find(name);
    if (it == engine->input_tensors.end()) return MNNR_ERROR_INVALID_PARAMETER;

    auto shape = it->second->shape();
    size_t ndims = shape.size() < 8 ? shape.size() : 8;
    *out_ndims = ndims;
    for (size_t i = 0; i < ndims; i++)
        dims[i] = static_cast<size_t>(shape[i]);
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_get_output_shape_by_name(
    const MNN_InferenceEngine *engine,
    const char *name,
    size_t *dims,
    size_t *out_ndims)
{
    if (!engine || !name || !dims || !out_ndims) return MNNR_ERROR_INVALID_PARAMETER;
    auto it = engine->output_tensors.find(name);
    if (it == engine->output_tensors.end()) return MNNR_ERROR_INVALID_PARAMETER;

    auto shape = it->second->shape();
    size_t ndims = shape.size() < 8 ? shape.size() : 8;
    *out_ndims = ndims;
    for (size_t i = 0; i < ndims; i++)
        dims[i] = static_cast<size_t>(shape[i]);
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_resize_input_by_name(
    MNN_InferenceEngine *engine,
    const char *name,
    const size_t *dims,
    size_t ndims)
{
    if (!engine || !name || !dims || ndims == 0)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> lock(engine->mutex);
    auto it = engine->input_tensors.find(name);
    if (it == engine->input_tensors.end())
    {
        engine->last_error = std::string("Input tensor not found: ") + name;
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::vector<int> new_shape(ndims);
    for (size_t i = 0; i < ndims; i++)
    {
        if (dims[i] == 0)
        {
            engine->last_error = std::string("Input tensor shape contains zero dimension for ") + name;
            return MNNR_ERROR_INVALID_PARAMETER;
        }
        new_shape[i] = static_cast<int>(dims[i]);
    }

    engine->interpreter->resizeTensor(it->second, new_shape);
    engine->interpreter->resizeSession(engine->default_session);
    engine->input_tensors.clear();
    engine->input_names.clear();
    engine->output_tensors.clear();
    engine->output_names.clear();
    if (!init_engine_tensors(engine))
    {
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_run_inference(
    MNN_InferenceEngine *engine,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!engine || !input_data || !output_data || engine->input_names.empty() || engine->output_names.empty())
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Build single-element arrays for the multi-I/O path
    const char *in_name = engine->input_names[0].c_str();
    const char *out_name = engine->output_names[0].c_str();
    return mnnr_run_multi(engine,
        &in_name, &input_data, &input_size, 1,
        &out_name, &output_data, &output_size, 1);
}

// ============== Multi-I/O Inference API ==============

MNNR_ErrorCode mnnr_run_multi(
    MNN_InferenceEngine *engine,
    const char **input_names,
    const float **input_data,
    const size_t *input_sizes,
    size_t input_count,
    const char **output_names,
    float **output_data,
    const size_t *output_sizes,
    size_t output_count)
{
    if (!engine || !input_names || !input_data || !input_sizes ||
        !output_names || !output_data || !output_sizes)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Copy all inputs
    for (size_t i = 0; i < input_count; i++)
    {
        auto it = engine->input_tensors.find(input_names[i]);
        if (it == engine->input_tensors.end())
        {
            engine->last_error = std::string("Input tensor not found: ") + input_names[i];
            return MNNR_ERROR_INVALID_PARAMETER;
        }

        auto input_host = make_unique_ptr<MNN::Tensor>(it->second, MNN::Tensor::CAFFE);
        std::memcpy(input_host->host<float>(), input_data[i], input_sizes[i] * sizeof(float));
        it->second->copyFromHostTensor(input_host.get());
    }

    // Run
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Copy all outputs
    for (size_t i = 0; i < output_count; i++)
    {
        auto it = engine->output_tensors.find(output_names[i]);
        if (it == engine->output_tensors.end())
        {
            engine->last_error = std::string("Output tensor not found: ") + output_names[i];
            return MNNR_ERROR_INVALID_PARAMETER;
        }

        auto output_host = make_unique_ptr<MNN::Tensor>(it->second, MNN::Tensor::CAFFE);
        it->second->copyToHostTensor(output_host.get());
        std::memcpy(output_data[i], output_host->host<float>(), output_sizes[i] * sizeof(float));
    }

    return MNNR_SUCCESS;
}

// ============== Per-Tensor Copy + Run API ==============

MNNR_ErrorCode mnnr_copy_input_f32(MNN_InferenceEngine *engine, const char *name, const float *data, size_t size)
{
    return copy_input_typed<float>(engine, name, data, size);
}

MNNR_ErrorCode mnnr_copy_input_i32(MNN_InferenceEngine *engine, const char *name, const int32_t *data, size_t size)
{
    return copy_input_typed<int32_t>(engine, name, data, size);
}

MNNR_ErrorCode mnnr_copy_input_i64(MNN_InferenceEngine *engine, const char *name, const int64_t *data, size_t size)
{
    return copy_input_typed<int64_t>(engine, name, data, size);
}

MNNR_ErrorCode mnnr_run(MNN_InferenceEngine *engine)
{
    if (!engine) return MNNR_ERROR_INVALID_PARAMETER;
    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    return MNNR_SUCCESS;
}

MNNR_ErrorCode mnnr_copy_output_f32(MNN_InferenceEngine *engine, const char *name, float *data, size_t size)
{
    return copy_output_typed<float>(engine, name, data, size);
}

MNNR_ErrorCode mnnr_copy_output_i32(MNN_InferenceEngine *engine, const char *name, int32_t *data, size_t size)
{
    return copy_output_typed<int32_t>(engine, name, data, size);
}

MNNR_ErrorCode mnnr_copy_output_i64(MNN_InferenceEngine *engine, const char *name, int64_t *data, size_t size)
{
    return copy_output_typed<int64_t>(engine, name, data, size);
}

int mnnr_get_input_dtype_at(const MNN_InferenceEngine *engine, size_t index)
{
    if (!engine || index >= engine->input_names.size()) return -1;
    auto it = engine->input_tensors.find(engine->input_names[index]);
    if (it == engine->input_tensors.end()) return -1;
    auto t = it->second->getType();
    return (int)t.code | ((int)t.bits << 16);
}

int mnnr_get_output_dtype_at(const MNN_InferenceEngine *engine, size_t index)
{
    if (!engine || index >= engine->output_names.size()) return -1;
    auto it = engine->output_tensors.find(engine->output_names[index]);
    if (it == engine->output_tensors.end()) return -1;
    auto t = it->second->getType();
    return (int)t.code | ((int)t.bits << 16);
}

const char *mnnr_get_last_error(const MNN_InferenceEngine *engine)
{
    if (!engine)
    {
        return "Engine is null";
    }
    return engine->last_error.c_str();
}

// ============== Session Pool API ==============

MNN_SessionPool *mnnr_create_session_pool(
    MNN_InferenceEngine *engine,
    size_t pool_size,
    const MNNR_Config *config)
{
    if (!engine || pool_size == 0)
    {
        return nullptr;
    }

    auto pool = new MNN_SessionPool();
    pool->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);

    // Create sessions
    for (size_t i = 0; i < pool_size; i++)
    {
        MNN::Session *session = engine->interpreter->createSession(schedule);
        if (!session)
        {
            // Cleanup on failure
            for (auto s : pool->sessions)
            {
                engine->interpreter->releaseSession(s);
            }
            // Note: input/output tensors are owned by MNN sessions, not by us.
            // They will be freed when sessions are released above.
            delete pool;
            return nullptr;
        }

        pool->sessions.push_back(session);
        pool->available_sessions.push(i);

        // Get input/output tensors for this session
        auto input_map = engine->interpreter->getSessionInputAll(session);
        auto output_map = engine->interpreter->getSessionOutputAll(session);

        pool->input_tensors.push_back(input_map.begin()->second);
        pool->output_tensors.push_back(output_map.begin()->second);
    }

    return pool;
}

void mnnr_destroy_session_pool(MNN_SessionPool *pool)
{
    if (pool)
    {
        for (auto session : pool->sessions)
        {
            if (pool->engine && pool->engine->interpreter)
            {
                pool->engine->interpreter->releaseSession(session);
            }
        }
        delete pool;
    }
}

MNNR_ErrorCode mnnr_session_pool_run(
    MNN_SessionPool *pool,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!pool || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Acquire a session (this will block if all sessions are busy)
    size_t session_idx;
    {
        std::unique_lock<std::mutex> lock(pool->mutex);
        pool->cv.wait(lock, [pool]
                      { return !pool->available_sessions.empty(); });
        session_idx = pool->available_sessions.front();
        pool->available_sessions.pop();
    }

    // Run inference with global lock to serialize MNN thread pool access
    MNNR_ErrorCode result = MNNR_SUCCESS;

    auto *session = pool->sessions[session_idx];
    auto *input_tensor = pool->input_tensors[session_idx];
    auto *output_tensor = pool->output_tensors[session_idx];

    // Create host tensor and copy input (can be done outside the global lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = pool->engine->interpreter->runSession(session);
        if (code != MNN::NO_ERROR)
        {
            pool->last_error = "Session pool inference failed";
            result = MNNR_ERROR_RUNTIME_ERROR;
        }
        else
        {
            // Copy output
            auto output_host = make_unique_ptr<MNN::Tensor>(output_tensor, MNN::Tensor::CAFFE);
            output_tensor->copyToHostTensor(output_host.get());
            std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
        }
    }

    // Release session
    {
        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->available_sessions.push(session_idx);
    }
    pool->cv.notify_one();

    return result;
}

size_t mnnr_session_pool_available(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return 0;
    }
    std::lock_guard<std::mutex> lock(const_cast<MNN_SessionPool *>(pool)->mutex);
    return pool->available_sessions.size();
}

const char *mnnr_session_pool_get_last_error(const MNN_SessionPool *pool)
{
    if (!pool)
    {
        return "Pool is null";
    }
    return pool->last_error.c_str();
}

// ============== Single Session API ==============

MNN_SingleSession *mnnr_create_session(
    MNN_InferenceEngine *engine,
    const MNNR_Config *config)
{
    if (!engine)
    {
        return nullptr;
    }

    auto session = new MNN_SingleSession();
    session->engine = engine;

    MNN::ScheduleConfig schedule;
    MNN::BackendConfig backend;
    init_schedule_config(schedule, backend, config);
    session->session = engine->interpreter->createSession(schedule);

    if (!session->session)
    {
        delete session;
        return nullptr;
    }

    // Get tensors
    auto input_map = engine->interpreter->getSessionInputAll(session->session);
    auto output_map = engine->interpreter->getSessionOutputAll(session->session);

    if (input_map.empty() || output_map.empty())
    {
        engine->interpreter->releaseSession(session->session);
        delete session;
        return nullptr;
    }

    session->input_tensor = input_map.begin()->second;
    session->output_tensor = output_map.begin()->second;

    return session;
}

void mnnr_destroy_session(MNN_SingleSession *session)
{
    if (session)
    {
        if (session->session && session->engine && session->engine->interpreter)
        {
            session->engine->interpreter->releaseSession(session->session);
        }
        delete session;
    }
}

MNNR_ErrorCode mnnr_run_inference_with_session(
    MNN_SingleSession *session,
    const float *input_data,
    size_t input_size,
    float *output_data,
    size_t output_size)
{
    if (!session || !input_data || !output_data)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    // Create host tensor and copy input (outside lock)
    auto input_host = make_unique_ptr<MNN::Tensor>(session->input_tensor, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, input_size * sizeof(float));

    {
        // Global lock for MNN inference to avoid thread pool exhaustion
        std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);

        session->input_tensor->copyFromHostTensor(input_host.get());

        // Run inference
        MNN::ErrorCode code = session->engine->interpreter->runSession(session->session);
        if (code != MNN::NO_ERROR)
        {
            session->last_error = "Session inference failed";
            return MNNR_ERROR_RUNTIME_ERROR;
        }

        // Copy output
        auto output_host = make_unique_ptr<MNN::Tensor>(session->output_tensor, MNN::Tensor::CAFFE);
        session->output_tensor->copyToHostTensor(output_host.get());
        std::memcpy(output_data, output_host->host<float>(), output_size * sizeof(float));
    }

    return MNNR_SUCCESS;
}

const char *mnnr_session_get_last_error(const MNN_SingleSession *session)
{
    if (!session)
    {
        return "Session is null";
    }
    return session->last_error.c_str();
}

// ============== Dynamic Shape API ==============

MNNR_ErrorCode mnnr_run_inference_dynamic(
    MNN_InferenceEngine *engine,
    const float *input_data,
    const size_t *input_dims,
    size_t input_ndims,
    float **output_data,
    size_t *output_size,
    size_t *output_dims,
    size_t *output_ndims)
{
    if (!engine || !input_data || !input_dims || !output_data || !output_size || !output_dims || !output_ndims)
    {
        return MNNR_ERROR_INVALID_PARAMETER;
    }

    std::lock_guard<std::mutex> global_lock(g_mnn_inference_mutex);
    std::lock_guard<std::mutex> lock(engine->mutex);

    // Build new input shape
    std::vector<int> new_shape(input_ndims);
    size_t total_input_size = 1;
    for (size_t i = 0; i < input_ndims; i++)
    {
        new_shape[i] = static_cast<int>(input_dims[i]);
        total_input_size *= input_dims[i];
    }

    // Resize input tensor (use first input)
    if (engine->input_names.empty())
    {
        engine->last_error = "No input tensors";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    auto first_input = engine->input_tensors[engine->input_names[0]];
    engine->interpreter->resizeTensor(first_input, new_shape);
    engine->interpreter->resizeSession(engine->default_session);

    // Get the updated input tensors after resize
    auto input_map = engine->interpreter->getSessionInputAll(engine->default_session);
    if (input_map.empty())
    {
        engine->last_error = "No input tensors found after resize";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->input_tensors.clear();
    engine->input_names.clear();
    for (auto &kv : input_map)
    {
        engine->input_tensors[kv.first] = kv.second;
        engine->input_names.push_back(kv.first);
    }
    first_input = engine->input_tensors[engine->input_names[0]];

    // Create host tensor and copy input data
    auto input_host = make_unique_ptr<MNN::Tensor>(first_input, MNN::Tensor::CAFFE);
    std::memcpy(input_host->host<float>(), input_data, total_input_size * sizeof(float));
    first_input->copyFromHostTensor(input_host.get());

    // Run inference
    MNN::ErrorCode code = engine->interpreter->runSession(engine->default_session);
    if (code != MNN::NO_ERROR)
    {
        engine->last_error = "Dynamic inference failed";
        return MNNR_ERROR_RUNTIME_ERROR;
    }

    // Get output tensors after inference
    auto output_map = engine->interpreter->getSessionOutputAll(engine->default_session);
    if (output_map.empty())
    {
        engine->last_error = "No output tensors found";
        return MNNR_ERROR_RUNTIME_ERROR;
    }
    engine->output_tensors.clear();
    engine->output_names.clear();
    for (auto &kv : output_map)
    {
        engine->output_tensors[kv.first] = kv.second;
        engine->output_names.push_back(kv.first);
    }
    auto first_output = engine->output_tensors[engine->output_names[0]];

    // Get output shape
    auto output_shape = first_output->shape();
    *output_ndims = output_shape.size();
    size_t total_output_size = 1;
    for (size_t i = 0; i < output_shape.size() && i < 8; i++)
    {
        output_dims[i] = static_cast<size_t>(output_shape[i]);
        total_output_size *= output_shape[i];
    }
    *output_size = total_output_size;

    // Allocate output buffer
    *output_data = new float[total_output_size];

    // Copy output data
    auto output_host = make_unique_ptr<MNN::Tensor>(first_output, MNN::Tensor::CAFFE);
    first_output->copyToHostTensor(output_host.get());
    std::memcpy(*output_data, output_host->host<float>(), total_output_size * sizeof(float));

    return MNNR_SUCCESS;
}

void mnnr_free_output(float *output_data)
{
    delete[] output_data;
}
