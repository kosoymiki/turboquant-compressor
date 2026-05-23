/**
 * TurboQuant OpenCL-Intercept-Layer — Kernel Execution Tracking
 */

#include "../include/tq_intercept.h"
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static pthread_mutex_t g_kernel_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct kernel_node {
    cl_kernel kernel;
    cl_program program;
    char kernel_name[256];
    cl_uint work_dim;
    size_t global_work_size[3];
    size_t local_work_size[3];
    uint64_t enqueue_time_ns;
    uint64_t start_time_ns;
    uint64_t end_time_ns;
    cl_event event;
    cl_int status;
    struct kernel_node* next;
} kernel_node_t;

typedef struct kernel_stats {
    uint64_t total_enqueues;
    uint64_t total_duration_ns;
    uint64_t min_duration_ns;
    uint64_t max_duration_ns;
    double avg_duration_ns;
    uint64_t total_global_work_items;
} kernel_stats_t;

static kernel_node_t* g_kernel_list = NULL;
static uint32_t g_kernel_count = 0;
static kernel_stats_t g_kernel_stats = {0};

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

void tq_intercept_record_kernel_enqueue(
    cl_kernel kernel, cl_program program, const char* kernel_name,
    cl_uint work_dim, const size_t* global_work_size, const size_t* local_work_size) {

    pthread_mutex_lock(&g_kernel_lock);

    kernel_node_t* node = (kernel_node_t*)malloc(sizeof(kernel_node_t));
    if (!node) {
        pthread_mutex_unlock(&g_kernel_lock);
        return;
    }

    memset(node, 0, sizeof(kernel_node_t));
    node->kernel = kernel;
    node->program = program;
    node->enqueue_time_ns = get_time_ns();
    node->start_time_ns = 0;
    node->end_time_ns = 0;
    node->work_dim = work_dim;

    if (kernel_name) {
        strncpy(node->kernel_name, kernel_name, sizeof(node->kernel_name) - 1);
    }

    if (global_work_size) {
        for (cl_uint i = 0; i < work_dim && i < 3; i++) {
            node->global_work_size[i] = global_work_size[i];
        }
    }

    if (local_work_size) {
        for (cl_uint i = 0; i < work_dim && i < 3; i++) {
            node->local_work_size[i] = local_work_size[i];
        }
    }

    node->next = g_kernel_list;
    g_kernel_list = node;
    g_kernel_count++;

    fprintf(stderr, "[TQ_KERNEL] enqueue: kernel=%s work_dim=%u global=[%zu,%zu,%zu]\n",
            node->kernel_name, work_dim,
            node->global_work_size[0], node->global_work_size[1],
            node->global_work_size[2]);

    g_kernel_stats.total_enqueues++;

    pthread_mutex_unlock(&g_kernel_lock);
}

void tq_intercept_record_kernel_start(cl_kernel kernel, cl_event event) {
    pthread_mutex_lock(&g_kernel_lock);

    kernel_node_t* curr = g_kernel_list;
    while (curr) {
        if (curr->kernel == kernel && curr->start_time_ns == 0) {
            curr->start_time_ns = get_time_ns();
            curr->event = event;
            fprintf(stderr, "[TQ_KERNEL] start: kernel=%s time=%lluns\n",
                    curr->kernel_name, (unsigned long long)curr->start_time_ns);
            break;
        }
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_kernel_lock);
}

void tq_intercept_record_kernel_complete(cl_kernel kernel, cl_event event, cl_int status) {
    pthread_mutex_lock(&g_kernel_lock);

    kernel_node_t* curr = g_kernel_list;
    while (curr) {
        if (curr->kernel == kernel && curr->end_time_ns == 0) {
            curr->end_time_ns = get_time_ns();
            curr->status = status;

            uint64_t duration = curr->end_time_ns - (curr->start_time_ns ? curr->start_time_ns : curr->enqueue_time_ns);

            fprintf(stderr, "[TQ_KERNEL] complete: kernel=%s duration=%lluns status=%d\n",
                    curr->kernel_name, (unsigned long long)duration, status);

            g_kernel_stats.total_duration_ns += duration;
            if (g_kernel_stats.min_duration_ns == 0 || duration < g_kernel_stats.min_duration_ns) {
                g_kernel_stats.min_duration_ns = duration;
            }
            if (duration > g_kernel_stats.max_duration_ns) {
                g_kernel_stats.max_duration_ns = duration;
            }

            size_t total_items = 1;
            for (cl_uint i = 0; i < curr->work_dim && i < 3; i++) {
                total_items *= curr->global_work_size[i] > 0 ? curr->global_work_size[i] : 1;
            }
            g_kernel_stats.total_global_work_items += total_items;

            if (g_kernel_stats.total_enqueues > 0) {
                g_kernel_stats.avg_duration_ns = (double)g_kernel_stats.total_duration_ns /
                                                g_kernel_stats.total_enqueues;
            }

            break;
        }
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_kernel_lock);
}

void tq_intercept_dump_kernel_tracking() {
    pthread_mutex_lock(&g_kernel_lock);

    fprintf(stderr, "[TQ_KERNEL] === Kernel Tracking Dump ===\n");
    fprintf(stderr, "  Total enqueues: %llu\n", (unsigned long long)g_kernel_stats.total_enqueues);
    fprintf(stderr, "  Total duration: %lluns\n", (unsigned long long)g_kernel_stats.total_duration_ns);
    fprintf(stderr, "  Avg duration: %.2fns\n", g_kernel_stats.avg_duration_ns);
    fprintf(stderr, "  Min duration: %lluns\n", (unsigned long long)g_kernel_stats.min_duration_ns);
    fprintf(stderr, "  Max duration: %lluns\n", (unsigned long long)g_kernel_stats.max_duration_ns);
    fprintf(stderr, "  Total work items: %llu\n", (unsigned long long)g_kernel_stats.total_global_work_items);

    kernel_node_t* curr = g_kernel_list;
    int idx = 0;
    while (curr && idx < 32) {
        uint64_t duration = 0;
        if (curr->end_time_ns > 0) {
            duration = curr->end_time_ns - (curr->start_time_ns ? curr->start_time_ns : curr->enqueue_time_ns);
        }
        fprintf(stderr, "  [%d] %s enqueue=%llu duration=%lluns status=%d\n",
                idx, curr->kernel_name, (unsigned long long)curr->enqueue_time_ns,
                (unsigned long long)duration, curr->status);
        curr = curr->next;
        idx++;
    }

    pthread_mutex_unlock(&g_kernel_lock);
}

void tq_intercept_get_kernel_stats(kernel_stats_t* stats) {
    if (!stats) return;

    pthread_mutex_lock(&g_kernel_lock);
    memcpy(stats, &g_kernel_stats, sizeof(kernel_stats_t));
    pthread_mutex_unlock(&g_kernel_lock);
}

void tq_intercept_clear_kernel_tracking() {
    pthread_mutex_lock(&g_kernel_lock);

    kernel_node_t* curr = g_kernel_list;
    while (curr) {
        kernel_node_t* next = curr->next;
        free(curr);
        curr = next;
    }
    g_kernel_list = NULL;
    g_kernel_count = 0;
    memset(&g_kernel_stats, 0, sizeof(kernel_stats_t));

    pthread_mutex_unlock(&g_kernel_lock);
}

const char* tq_intercept_kernel_name_from_handle(cl_kernel kernel) {
    pthread_mutex_lock(&g_kernel_lock);

    kernel_node_t* curr = g_kernel_list;
    while (curr) {
        if (curr->kernel == kernel) {
            pthread_mutex_unlock(&g_kernel_lock);
            return curr->kernel_name;
        }
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_kernel_lock);
    return "unknown";
}