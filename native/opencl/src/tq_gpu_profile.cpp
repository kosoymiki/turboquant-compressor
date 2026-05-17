/**
 * TurboQuant GPU Auto-Profile — implementation.
 *
 * GPU detection via KGSL ioctl + renderer string fallback.
 * Profile database covers all Adreno 6xx/7xx/8xx series.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "../include/tq_gpu_profile.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <mutex>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/stat.h>

namespace tq {

// KGSL ioctl (same as tq_driver_loader.cpp)
#define KGSL_IOC_TYPE 0x09
#define IOCTL_KGSL_DEVICE_GETPROPERTY _IOWR(KGSL_IOC_TYPE, 0x02, struct kgsl_device_getproperty)
#define KGSL_PROP_DEVICE_INFO 0x1

struct kgsl_device_getproperty {
    unsigned int type;
    void* value;
    size_t sizebytes;
};

struct kgsl_devinfo {
    unsigned int device_id;
    unsigned int chip_id;
    unsigned int mmu_enabled;
    unsigned long gmem_gpubaseaddr;
    unsigned int gpu_id;
    size_t gmem_sizebytes;
};

static uint32_t kgsl_get_chip_id() {
    int fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return 0;

    kgsl_devinfo info = {};
    kgsl_device_getproperty prop = {
        .type = KGSL_PROP_DEVICE_INFO,
        .value = &info,
        .sizebytes = sizeof(info),
    };

    int ret;
    do { ret = ioctl(fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &prop); } while (ret == -1 && errno == EINTR);
    close(fd);
    return (ret == 0) ? info.chip_id : 0;
}

static size_t kgsl_get_gmem_size() {
    int fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC);
    if (fd < 0) return 0;

    kgsl_devinfo info = {};
    kgsl_device_getproperty prop = {
        .type = KGSL_PROP_DEVICE_INFO,
        .value = &info,
        .sizebytes = sizeof(info),
    };

    int ret;
    do { ret = ioctl(fd, IOCTL_KGSL_DEVICE_GETPROPERTY, &prop); } while (ret == -1 && errno == EINTR);
    close(fd);
    return (ret == 0) ? info.gmem_sizebytes : 0;
}

// --- Profile database ---

static GpuProfile make_default() {
    return {
        .gen = GpuGen::UNKNOWN,
        .chip_id = 0,
        .name = "Unknown GPU",
        .compute_wave = WaveMode::WAVE64,
        .preferred_wg_size = 256,
        .max_waves_per_cu = 16,
        .reg_size_vec4 = 64,
        .gmem_size_kb = 512,
        .pc_mode_cntl = 0x3f,
        .tile_strategy = TileStrategy::AUTO,
        .supports_fp16_fma = false,
        .supports_dot_product = false,
        .dual_wave_dispatch = false,
        .attention_wg_size = 256,
        .dequant_wg_size = 256,
        .mse_wg_size = 256,
        .qjl_wg_size = 256,
    };
}

GpuProfile profile_from_chip_id(uint32_t chip_id) {
    GpuProfile p = make_default();
    size_t gmem = kgsl_get_gmem_size();
    if (gmem > 0) p.gmem_size_kb = gmem / 1024;

    // A6XX series
    if ((chip_id >> 24) == 0x06 || (chip_id & 0xFF000000) == 0x06000000) {
        p.gen = GpuGen::ADRENO_6XX;
        p.name = "Adreno 6xx";
        p.compute_wave = WaveMode::WAVE64;
        p.preferred_wg_size = 128;
        p.reg_size_vec4 = 64;
        p.pc_mode_cntl = 0x3f;
        p.tile_strategy = TileStrategy::GMEM_TILED;
        p.supports_fp16_fma = true;
        p.supports_dot_product = false;
        p.attention_wg_size = 128;
        p.dequant_wg_size = 128;
        p.mse_wg_size = 128;
        p.qjl_wg_size = 64;
        return p;
    }

    // A7XX series — detect specific models
    uint8_t top = (chip_id >> 24) & 0xFF;
    if (top == 0x07 || (chip_id & 0xFFFF0000) == 0x43050000 ||
        (chip_id & 0xFFFF0000) == 0x43030000) {

        p.gen = GpuGen::ADRENO_7XX;
        p.compute_wave = WaveMode::WAVE128;
        p.reg_size_vec4 = 64;
        p.supports_fp16_fma = true;
        p.supports_dot_product = true;
        p.tile_strategy = TileStrategy::GMEM_TILED;

        // A730/A725: chip_id 0x43050a01 / 0x43050c01
        if ((chip_id & 0xFFFF0000) == 0x43050000) {
            p.name = "Adreno 730/725";
            p.chip_id = chip_id;
            p.pc_mode_cntl = 0x1f1f;  // 2x perf vs 0x3f (Mesa issue #15411)
            p.preferred_wg_size = 512;
            p.max_waves_per_cu = 16;
            p.attention_wg_size = 512;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
            return p;
        }

        // A740/A750: chip_id 0x43051401 (A740), 0x43050f01 (A750)
        if (chip_id == 0x43051401 || chip_id == 0x43050f01) {
            p.name = (chip_id == 0x43051401) ? "Adreno 740" : "Adreno 750";
            p.chip_id = chip_id;
            p.pc_mode_cntl = 0x1f1f;
            p.preferred_wg_size = 512;
            p.max_waves_per_cu = 16;
            p.attention_wg_size = 512;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
            return p;
        }

        // A710: chip_id 0x07010001
        if (chip_id == 0x07010001) {
            p.name = "Adreno 710";
            p.chip_id = chip_id;
            p.pc_mode_cntl = 0x1f1f;
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 128;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
            return p;
        }

        // A720: chip_id 0x07030003
        if (chip_id == 0x07030003) {
            p.name = "Adreno 720";
            p.chip_id = chip_id;
            p.pc_mode_cntl = 0x1f1f;
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
            return p;
        }

        // A715: chip_id 0x43030c01
        if (chip_id == 0x43030c01) {
            p.name = "Adreno 715";
            p.chip_id = chip_id;
            p.pc_mode_cntl = 0x1f1f;
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
            return p;
        }

        // Generic A7XX fallback
        p.name = "Adreno 7xx (generic)";
        p.chip_id = chip_id;
        p.pc_mode_cntl = 0x1f1f;
        p.preferred_wg_size = 256;
        p.attention_wg_size = 256;
        p.dequant_wg_size = 256;
        p.mse_wg_size = 256;
        p.qjl_wg_size = 128;
        return p;
    }

    // A8XX series — dual wave dispatch
    // KGSL returns 32-bit chip_id; A8XX uses 0x4402xxxx / 0x4403xxxx / 0x4407xxxx
    uint16_t hi16 = (chip_id >> 16) & 0xFFFF;
    if (top == 0x08 || hi16 == 0x4402 || hi16 == 0x4403 || hi16 == 0x4407) {

        p.gen = GpuGen::ADRENO_8XX;
        p.compute_wave = WaveMode::WAVE64;  // A8XX uses dual wave64, NOT wave128
        p.dual_wave_dispatch = true;
        p.reg_size_vec4 = 96;
        p.supports_fp16_fma = true;
        p.supports_dot_product = true;
        p.pc_mode_cntl = 0x3f;
        p.tile_strategy = TileStrategy::GMEM_TILED;

        // A810: chip_id 0x4402xxxx
        if (hi16 == 0x4402) {
            p.name = "Adreno 810";
            p.chip_id = chip_id;
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
            return p;
        }

        // A830: chip_id 0x4403xxxx
        if (hi16 == 0x4403) {
            p.name = "Adreno 830";
            p.chip_id = chip_id;
            p.reg_size_vec4 = 128;
            p.preferred_wg_size = 512;
            p.attention_wg_size = 512;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
            return p;
        }

        // A850: chip_id 0x4407xxxx
        if (hi16 == 0x4407) {
            p.name = "Adreno 850";
            p.chip_id = chip_id;
            p.reg_size_vec4 = 128;
            p.preferred_wg_size = 1024;
            p.attention_wg_size = 1024;
            p.dequant_wg_size = 512;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
            return p;
        }

        // Generic A8XX
        p.name = "Adreno 8xx (generic)";
        p.chip_id = chip_id;
        p.preferred_wg_size = 256;
        p.attention_wg_size = 256;
        p.dequant_wg_size = 256;
        p.mse_wg_size = 256;
        p.qjl_wg_size = 128;
        return p;
    }

    return p;
}

GpuProfile profile_from_renderer(const char* renderer) {
    GpuProfile p = make_default();
    if (!renderer) return p;

    // Match "Adreno (TM) NNN" or "Adreno NNN"
    const char* adreno = strstr(renderer, "dreno");
    if (!adreno) return p;

    // Find the 3-digit model number
    const char* ptr = adreno;
    int model = 0;
    while (*ptr) {
        if (*ptr >= '0' && *ptr <= '9') {
            model = 0;
            while (*ptr >= '0' && *ptr <= '9') {
                model = model * 10 + (*ptr - '0');
                ptr++;
            }
            if (model >= 600 && model <= 899) break;
            model = 0;
        }
        ptr++;
    }

    if (model >= 600 && model < 700) {
        p.gen = GpuGen::ADRENO_6XX;
        p.name = "Adreno " + std::to_string(model);
        p.compute_wave = WaveMode::WAVE64;
        p.preferred_wg_size = 128;
        p.reg_size_vec4 = 64;
        p.supports_fp16_fma = true;
        p.tile_strategy = TileStrategy::GMEM_TILED;
        p.attention_wg_size = 128;
        p.dequant_wg_size = 128;
        p.mse_wg_size = 128;
        p.qjl_wg_size = 64;
    } else if (model >= 700 && model < 800) {
        p.gen = GpuGen::ADRENO_7XX;
        p.name = "Adreno " + std::to_string(model);
        p.compute_wave = WaveMode::WAVE128;
        p.reg_size_vec4 = 64;
        p.pc_mode_cntl = 0x1f1f;
        p.supports_fp16_fma = true;
        p.supports_dot_product = true;
        p.tile_strategy = TileStrategy::GMEM_TILED;
        // A730/740/750 get larger WG
        if (model >= 730) {
            p.preferred_wg_size = 512;
            p.attention_wg_size = 512;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
        } else {
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
        }
    } else if (model >= 800 && model < 900) {
        p.gen = GpuGen::ADRENO_8XX;
        p.name = "Adreno " + std::to_string(model);
        p.compute_wave = WaveMode::WAVE64;
        p.dual_wave_dispatch = true;
        p.reg_size_vec4 = (model >= 830) ? 128 : 96;
        p.supports_fp16_fma = true;
        p.supports_dot_product = true;
        p.tile_strategy = TileStrategy::GMEM_TILED;
        if (model >= 850) {
            p.preferred_wg_size = 1024;
            p.attention_wg_size = 1024;
            p.dequant_wg_size = 512;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
        } else if (model >= 830) {
            p.preferred_wg_size = 512;
            p.attention_wg_size = 512;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 512;
            p.qjl_wg_size = 256;
        } else {
            p.preferred_wg_size = 256;
            p.attention_wg_size = 256;
            p.dequant_wg_size = 256;
            p.mse_wg_size = 256;
            p.qjl_wg_size = 128;
        }
    }

    return p;
}

// Cached profile
static std::once_flag g_profile_once;
static GpuProfile g_profile;

GpuProfile detect_gpu_profile() {
    uint32_t chip_id = kgsl_get_chip_id();
    if (chip_id != 0) {
        return profile_from_chip_id(chip_id);
    }
    // Fallback: no KGSL access, return generic
    return make_default();
}

const GpuProfile& active_profile() {
    std::call_once(g_profile_once, []() {
        g_profile = detect_gpu_profile();
    });
    return g_profile;
}

std::string get_build_opts(const GpuProfile& p) {
    std::string opts = "-cl-std=CL3.0 -cl-mad-enable -cl-fast-relaxed-math";

    if (p.supports_fp16_fma) {
        opts += " -DTQ_HAS_FP16=1";
    }
    if (p.supports_dot_product) {
        opts += " -DTQ_HAS_DOT4=1";
    }

    // Wave size define for kernel specialization
    opts += " -DTQ_WAVE_SIZE=" + std::to_string((int)p.compute_wave);

    // Workgroup size hints
    opts += " -DTQ_ATTENTION_WG=" + std::to_string(p.attention_wg_size);
    opts += " -DTQ_DEQUANT_WG=" + std::to_string(p.dequant_wg_size);
    opts += " -DTQ_MSE_WG=" + std::to_string(p.mse_wg_size);
    opts += " -DTQ_QJL_WG=" + std::to_string(p.qjl_wg_size);

    // Gen-specific
    if (p.gen == GpuGen::ADRENO_7XX) {
        opts += " -DTQ_GEN7=1";
    } else if (p.gen == GpuGen::ADRENO_8XX) {
        opts += " -DTQ_GEN8=1 -DTQ_DUAL_WAVE=1";
    }

    // Register pressure hint
    opts += " -DTQ_REG_VEC4=" + std::to_string(p.reg_size_vec4);

    return opts;
}

} // namespace tq
