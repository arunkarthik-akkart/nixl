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
DEFINE_string(backend, XFERBENCH_BACKEND_UCX, "Name of communication backend [UCX, UCX_MO, GDS, POSIX] \
              (only used with nixl worker)");
DEFINE_string(op_type, XFERBENCH_OP_WRITE, "Op type: READ, WRITE");
DEFINE_uint64(total_buffer_size, 8LL * 1024 * (1 << 20), "Total buffer \
              size across device for each process (Default: 80 GiB)");
DEFINE_int32(num_threads, 1,
             "Number of threads used by benchmark."
             " Num_iter must be greater or equal than num_threads and equally divisible by num_threads."
             " (Default: 1)");
DEFINE_bool(enable_pt, false, "Enable Progress Thread (only used with nixl worker)");

// TODO: We should take rank wise device list as input to extend support
// <rank>:<device_list>, ...
// For example- 0:mlx5_0,mlx5_1,mlx5_2,1:mlx5_3,mlx5_4, ...
DEFINE_string(device_list, "all", "Comma-separated device name to use for \
		      communication (only used with nixl worker)");
DEFINE_string(etcd_endpoints, "http://localhost:2379", "ETCD server endpoints for communication");

std::string xferBenchConfig::backend = "";
std::string xferBenchConfig::op_type = "";
size_t xferBenchConfig::total_buffer_size = 0;
int xferBenchConfig::num_threads = 0;
bool xferBenchConfig::enable_pt = false;
std::string xferBenchConfig::device_list = "";
std::string xferBenchConfig::etcd_endpoints = "";
std::vector<std::string> devices = { };

int xferBenchConfig::loadFromFlags() {
    backend = FLAGS_backend;
    enable_pt = FLAGS_enable_pt;
    device_list = FLAGS_device_list;
    op_type = FLAGS_op_type;
    total_buffer_size = FLAGS_total_buffer_size;
    num_threads = FLAGS_num_threads;
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
    std::cout << std::left << std::setw(60) << "Device list (--device_list=dev1,dev2,...)" << ": "
                << device_list << std::endl;
    std::cout << std::left << std::setw(60) << "Op type (--op_type=[READ,WRITE])" << ": "
              << op_type << std::endl;
    std::cout << std::left << std::setw(60) << "Total buffer size (--total_buffer_size=N)" << ": "
              << total_buffer_size << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::endl;
}

std::vector<std::string> xferBenchConfig::parseDeviceList() {
    std::vector<std::string> devices;
    std::string dev;
    std::stringstream ss(xferBenchConfig::device_list);
    devices.push_back("all");

    return devices;
}

/**********
 * xferBench Utils
 **********/
xferBenchRT *xferBenchUtils::rt = nullptr;
std::string xferBenchUtils::dev_to_use = "";

void xferBenchUtils::setRT(xferBenchRT *rt) {
    xferBenchUtils::rt = rt;
}

void xferBenchUtils::setDevToUse(std::string dev) {
    dev_to_use = dev;
}

std::string xferBenchUtils::getDevToUse() {
    return dev_to_use;
}

void xferBenchUtils::printStatsHeader() {
    std::cout << std::left << std::setw(20) << "Block Size (B)"
                << std::setw(15) << "Batch Size"
                << std::setw(15) << "Avg Lat. (us)"
                << std::setw(15) << "B/W (MiB/Sec)"
                << std::setw(15) << "B/W (GiB/Sec)"
                << std::setw(15) << "B/W (GB/Sec)"
                << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void xferBenchUtils::printStats(bool is_target, size_t block_size, size_t batch_size, double total_duration) {
    size_t total_data_transferred = 0;
    double avg_latency = 0, throughput = 0, throughput_gib = 0, throughput_gb = 0;

    total_data_transferred = ((block_size * batch_size) * 1); // In Bytes
    avg_latency = (total_duration / (1 * batch_size)); // In microsec

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
                << std::setw(15) << batch_size
                << std::setw(15) << avg_latency
                << std::setw(15) << throughput
                << std::setw(15) << throughput_gib
                << std::setw(15) << throughput_gb
                << std::endl;
}
