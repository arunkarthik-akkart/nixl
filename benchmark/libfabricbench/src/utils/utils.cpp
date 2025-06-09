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

#include <cstring>
#include <gflags/gflags.h>
#include <sstream>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include <iomanip>
#include <omp.h>
#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "runtime/etcd/etcd_rt.h"
#include "utils/utils.h"


/**********
 * xferBench Config
 **********/
DEFINE_string(backend, XFERBENCH_BACKEND_UCX, "Name of communication backend [UCX] \
              (only used with nixl worker)");
DEFINE_string(op_type, XFERBENCH_OP_WRITE, "Op type: READ, WRITE");
DEFINE_bool(enable_pt, false, "Enable Progress Thread (only used with nixl worker)");

DEFINE_string(etcd_endpoints, "http://localhost:2379", "ETCD server endpoints for communication");

std::string xferBenchConfig::backend = "";
std::string xferBenchConfig::op_type = "";
bool xferBenchConfig::enable_pt = false;
std::string xferBenchConfig::etcd_endpoints = "";
std::vector<std::string> devices = { };

int xferBenchConfig::loadFromFlags() {
    backend = FLAGS_backend;
    enable_pt = FLAGS_enable_pt;
    op_type = FLAGS_op_type;
    etcd_endpoints = FLAGS_etcd_endpoints;
    return 0;
}

void xferBenchConfig::printConfig() {
    std::cout << std::string(70, '*') << std::endl;
    std::cout << "NIXLBench Configuration" << std::endl;
    std::cout << std::string(70, '*') << std::endl;
    std::cout << std::left << std::setw(60) << "Backend (--backend=[UCX])" << ": "
                << backend << std::endl;
    std::cout << std::left << std::setw(60) << "Enable pt (--enable_pt=[0,1])" << ": "
                << enable_pt << std::endl;
    std::cout << std::left << std::setw(60) << "Op type (--op_type=[READ,WRITE])" << ": "
              << op_type << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::endl;
}

/**********
 * xferBench Utils
 **********/
xferBenchRT *xferBenchUtils::rt = nullptr;

void xferBenchUtils::setRT(xferBenchRT *rt) {
    xferBenchUtils::rt = rt;
}

void xferBenchUtils::printStatsHeader() {
    std::cout << std::left << std::setw(20) << "Block Size (B)"
                << std::setw(15) << "Avg Lat. (us)"
                << std::setw(15) << "B/W (MiB/Sec)"
                << std::setw(15) << "B/W (GiB/Sec)"
                << std::setw(15) << "B/W (GB/Sec)"
                << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void xferBenchUtils::printStats(bool is_target, size_t block_size, double total_duration) {
    size_t total_data_transferred = 0;
    double avg_latency = 0, throughput = 0, throughput_gib = 0, throughput_gb = 0;

    total_data_transferred = block_size; // In Bytes
    avg_latency = total_duration; // In microsec

    throughput = (((double) total_data_transferred / (1024 * 1024)) /
                   (total_duration / 1e6));   // In MiB/Sec
    throughput_gib = (throughput / 1024);   // In GiB/Sec
    throughput_gb = (((double) total_data_transferred / (1000 * 1000 * 1000)) /
                   (total_duration / 1e6));   // In GB/Sec

    if (rt->getRank() != 0) {
        return;
    }

    // Tabulate print with fixed width for each string
    std::cout << std::left << std::setw(20) << block_size
                << std::setw(15) << avg_latency
                << std::setw(15) << throughput
                << std::setw(15) << throughput_gib
                << std::setw(15) << throughput_gb
                << std::endl;
}
