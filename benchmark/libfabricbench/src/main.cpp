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

#include "config.h"
#include <iostream>
#include <nixl.h>
#include <sys/time.h>
#include <gflags/gflags.h>
#include "utils/utils.h"
#include "utils/scope_guard.h"
#include "worker/nixl/nixl_worker.h"

#include <unistd.h>
#include <memory>
#include <csignal>

static std::vector<std::vector<xferBenchIOV>> createTransferDescLists(xferBenchWorker &worker,
                                                                      std::vector<std::vector<xferBenchIOV>> &iov_lists,
                                                                      size_t block_size,
                                                                      size_t batch_size,
                                                                      int num_threads) {

    size_t count = 1;
    size_t stride = xferBenchConfig::total_buffer_size / (num_threads);

    std::vector<std::vector<xferBenchIOV>> xfer_lists;

    for (const auto &iov_list: iov_lists) {
        std::vector<xferBenchIOV> xfer_list;

        for (const auto &iov : iov_list) {
            for (size_t i = 0; i < count; i++) {
                size_t dev_offset = ((i * stride) % iov.len);

                for (size_t j = 0; j < batch_size; j++) {
                    size_t block_offset = ((j * block_size) % iov.len);
                    xfer_list.push_back(xferBenchIOV((iov.addr + dev_offset) + block_offset,
                                                      block_size,
                                                      iov.devId));
                }
            }
        }

        xfer_lists.push_back(xfer_list);
    }

    return xfer_lists;
}

static int processBatchSizes(xferBenchWorker &worker,
                             std::vector<std::vector<xferBenchIOV>> &iov_lists,
                             size_t block_size, int num_threads) {
    size_t batch_size = 1;

    auto local_trans_lists = createTransferDescLists(worker,
                                                        iov_lists,
                                                        block_size,
                                                        batch_size,
                                                        num_threads);

    if (worker.isTarget()) {
        worker.exchangeIOV(local_trans_lists);
        worker.poll(block_size);

    } else if (worker.isInitiator()) {
        std::vector<std::vector<xferBenchIOV>> remote_trans_lists(worker.exchangeIOV(local_trans_lists));

        auto result = worker.transfer(block_size,
                                        local_trans_lists,
                                        remote_trans_lists);
        if (std::holds_alternative<int>(result)) {
            return 1;
        }

        xferBenchUtils::printStats(false, block_size, batch_size,
                                std::get<double>(result));
    }

    return 0;
}

static std::unique_ptr<xferBenchWorker> createWorker(int *argc, char ***argv) {
    std::vector<std::string> devices = xferBenchConfig::parseDeviceList();
    if (devices.empty()) {
        std::cerr << "Failed to parse device list" << std::endl;
        return nullptr;
    }
    return std::make_unique<xferBenchNixlWorker>(argc, argv, devices);
}

int main(int argc, char *argv[]) {

    // Parse Command Line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    int ret = xferBenchConfig::loadFromFlags();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    int num_threads = xferBenchConfig::num_threads;

    // Create the appropriate worker based on worker configuration
    std::unique_ptr<xferBenchWorker> worker_ptr = createWorker(&argc, &argv);
    if (!worker_ptr) {
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, worker_ptr->signalHandler);

    // Ensure all processes are ready before exchanging metadata
    ret = worker_ptr->synchronizeStart();
    if (0 != ret) {
        std::cerr << "Failed to synchronize all processes" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<std::vector<xferBenchIOV>> iov_lists = worker_ptr->allocateMemory(num_threads);
    auto mem_guard = make_scope_guard ([&] {
        worker_ptr->deallocateMemory(iov_lists);
    });

    ret = worker_ptr->exchangeMetadata();
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    if (worker_ptr->isInitiator() && worker_ptr->isMasterRank()) {
        xferBenchConfig::printConfig();
        xferBenchUtils::printStatsHeader();
    }

    size_t block_size = 16*1024;

    ret = processBatchSizes(*worker_ptr, iov_lists, block_size, num_threads);
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    ret = worker_ptr->synchronize(); // Make sure environment is not used anymore
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    gflags::ShutDownCommandLineFlags();

    return worker_ptr->signaled() ? EXIT_FAILURE : EXIT_SUCCESS;
}
