/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __UTILS_H
#define __UTILS_H

#include "config.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include "runtime/runtime.h"

// TODO: This is true for CX-7, need support for other CX cards and NVLink
#define MAXBW 50.0 // 400 Gbps or 50 GB/sec
#define LARGE_BLOCK_SIZE (1LL * (1 << 20))
#define LARGE_BLOCK_SIZE_ITER_FACTOR 16

#define XFERBENCH_INITIATOR_BUFFER_ELEMENT 0xbb
#define XFERBENCH_TARGET_BUFFER_ELEMENT 0xaa

// Runtime types
#define XFERBENCH_RT_ETCD "ETCD"

// Backend types
#define XFERBENCH_BACKEND_UCX "UCX"

// Operation types
#define XFERBENCH_OP_READ  "READ"
#define XFERBENCH_OP_WRITE "WRITE"

class xferBenchConfig {
    public:
        static std::string worker_type;
        static std::string backend;
        static std::string op_type;
        static size_t total_buffer_size;
        static bool enable_pt;
        static std::string etcd_endpoints;
        static int loadFromFlags();
        static void printConfig();
};

// Generic IOV descriptor class independent of NIXL
class xferBenchIOV {
public:
    uintptr_t addr;
    size_t len;
    int devId;

    xferBenchIOV(uintptr_t a, size_t l, int d) : addr(a), len(l), devId(d) {}
};

class xferBenchUtils {
    private:
        static xferBenchRT *rt;
    public:
        static void setRT(xferBenchRT *rt);

        static void printStatsHeader();
        static void printStats(bool is_target, size_t block_size, size_t batch_size,
			                   double total_duration);
};

#endif // __UTILS_H
