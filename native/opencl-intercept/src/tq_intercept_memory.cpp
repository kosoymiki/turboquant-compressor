/**
 * TurboQuant OpenCL-Intercept-Layer — Memory Operations Tracking
 */

#include "../include/tq_intercept.h"
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

static pthread_mutex_t g_mem_lock = PTHREAD_MUTEX_INITIALIZER;

typedef struct mem_node {
    cl_mem mem;
    size_t size;
    cl_mem_object_type type;
    cl_mem_flags flags;
    void* host_ptr;
    cl_context context;
    uint64_t alloc_time_ns;
    uint64_t last_access_ns;
    uint32_t access_count;
    char kernel_name[128];
    struct mem_node* next;
} mem_node_t;

static mem_node_t* g_mem_list = NULL;
static uint32_t g_mem_count = 0;

static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static mem_node_t* find_mem_node(cl_mem mem) {
    mem_node_t* curr = g_mem_list;
    while (curr) {
        if (curr->mem == mem) return curr;
        curr = curr->next;
    }
    return NULL;
}

void tq_intercept_record_mem_alloc(
    cl_mem mem, size_t size, cl_mem_object_type type,
    cl_mem_flags flags, void* host_ptr, cl_context ctx) {

    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* node = (mem_node_t*)malloc(sizeof(mem_node_t));
    if (!node) {
        pthread_mutex_unlock(&g_mem_lock);
        return;
    }

    memset(node, 0, sizeof(mem_node_t));
    node->mem = mem;
    node->size = size;
    node->type = type;
    node->flags = flags;
    node->host_ptr = host_ptr;
    node->context = ctx;
    node->alloc_time_ns = get_time_ns();
    node->last_access_ns = node->alloc_time_ns;
    node->access_count = 0;
    node->next = g_mem_list;
    g_mem_list = node;
    g_mem_count++;

    fprintf(stderr, "[TQ_MEM] alloc: mem=%p size=%zu type=%d flags=0x%llx\n",
            (void*)mem, size, type, (unsigned long long)flags);

    pthread_mutex_unlock(&g_mem_lock);
}

void tq_intercept_record_mem_free(cl_mem mem) {
    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* prev = NULL;
    mem_node_t* curr = g_mem_list;

    while (curr) {
        if (curr->mem == mem) {
            if (prev) {
                prev->next = curr->next;
            } else {
                g_mem_list = curr->next;
            }
            fprintf(stderr, "[TQ_MEM] free: mem=%p size=%zu alloc_time=%lluns\n",
                    (void*)mem, curr->size, (unsigned long long)curr->alloc_time_ns);
            free(curr);
            g_mem_count--;
            break;
        }
        prev = curr;
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_mem_lock);
}

void tq_intercept_record_mem_access(cl_mem mem, const char* kernel_name) {
    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* node = find_mem_node(mem);
    if (node) {
        node->access_count++;
        node->last_access_ns = get_time_ns();
        if (kernel_name) {
            strncpy(node->kernel_name, kernel_name, sizeof(node->kernel_name) - 1);
        }
        fprintf(stderr, "[TQ_MEM] access: mem=%p kernel=%s access_count=%u\n",
                (void*)mem, kernel_name ? kernel_name : "unknown", node->access_count);
    }

    pthread_mutex_unlock(&g_mem_lock);
}

void tq_intercept_record_mem_read(
    cl_mem mem, size_t offset, size_t size, const char* kernel_name) {

    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* node = find_mem_node(mem);
    if (node) {
        node->access_count++;
        node->last_access_ns = get_time_ns();
        fprintf(stderr, "[TQ_MEM] READ: mem=%p offset=%zu size=%zu kernel=%s\n",
                (void*)mem, offset, size, kernel_name ? kernel_name : "unknown");
    }

    pthread_mutex_unlock(&g_mem_lock);
}

void tq_intercept_record_mem_write(
    cl_mem mem, size_t offset, size_t size, const char* kernel_name) {

    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* node = find_mem_node(mem);
    if (node) {
        node->access_count++;
        node->last_access_ns = get_time_ns();
        fprintf(stderr, "[TQ_MEM] WRITE: mem=%p offset=%zu size=%zu kernel=%s\n",
                (void*)mem, offset, size, kernel_name ? kernel_name : "unknown");
    }

    pthread_mutex_unlock(&g_mem_lock);
}

size_t tq_intercept_get_total_allocated_bytes() {
    pthread_mutex_lock(&g_mem_lock);

    size_t total = 0;
    mem_node_t* curr = g_mem_list;
    while (curr) {
        total += curr->size;
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_mem_lock);
    return total;
}

uint32_t tq_intercept_get_allocation_count() {
    pthread_mutex_lock(&g_mem_lock);
    uint32_t count = g_mem_count;
    pthread_mutex_unlock(&g_mem_lock);
    return count;
}

void tq_intercept_dump_mem_tracking() {
    pthread_mutex_lock(&g_mem_lock);

    fprintf(stderr, "[TQ_MEM] === Memory Tracking Dump ===\n");
    fprintf(stderr, "  Total allocations: %u\n", g_mem_count);
    fprintf(stderr, "  Total bytes: %zu\n", tq_intercept_get_total_allocated_bytes());

    mem_node_t* curr = g_mem_list;
    int idx = 0;
    while (curr) {
        fprintf(stderr, "  [%d] mem=%p size=%zu type=%d kernel=%s accesses=%u\n",
                idx++, (void*)curr->mem, curr->size, curr->type,
                curr->kernel_name[0] ? curr->kernel_name : "-", curr->access_count);
        curr = curr->next;
    }

    pthread_mutex_unlock(&g_mem_lock);
}

void tq_intercept_clear_mem_tracking() {
    pthread_mutex_lock(&g_mem_lock);

    mem_node_t* curr = g_mem_list;
    while (curr) {
        mem_node_t* next = curr->next;
        free(curr);
        curr = next;
    }
    g_mem_list = NULL;
    g_mem_count = 0;

    pthread_mutex_unlock(&g_mem_lock);
}