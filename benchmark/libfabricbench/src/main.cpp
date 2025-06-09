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


static int processBatchSizes(xferBenchWorker &worker,
                             xferBenchIOV &iov,
                             size_t block_size) {

    xferBenchIOV local_trans(iov.addr,block_size, iov.devId);

    if (worker.isTarget()) {
        worker.exchangeIOV(local_trans);
        worker.poll(block_size);

    } else if (worker.isInitiator()) {
        xferBenchIOV remote_trans(worker.exchangeIOV(local_trans));

        auto result = worker.transfer(block_size,
                                        local_trans,
                                        remote_trans);
        if (std::holds_alternative<int>(result)) {
            return 1;
        }

        xferBenchUtils::printStats(false, block_size, 1,
                                std::get<double>(result));
    }

    return 0;
}

static std::unique_ptr<xferBenchWorker> createWorker(int *argc, char ***argv) {
    return std::make_unique<xferBenchNixlWorker>(argc, argv);
}

int main(int argc, char *argv[]) {

    // Parse Command Line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    int ret = xferBenchConfig::loadFromFlags();
    if (0 != ret) {
        return EXIT_FAILURE;
    }
    std::cout << "Processed Command Line arguments" << std::endl;

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

    std::cout << "All processes are ready to start" << std::endl;

    xferBenchIOV iov = worker_ptr->allocateMemory();
    auto mem_guard = make_scope_guard ([&] {
        worker_ptr->deallocateMemory(iov);
    });

    ret = worker_ptr->exchangeMetadata();
    if (0 != ret) {
        return EXIT_FAILURE;
    }
    std::cout << "Memory Exchange metadata complete." << std::endl;

    if (worker_ptr->isInitiator() && worker_ptr->isMasterRank()) {
        xferBenchConfig::printConfig();
        xferBenchUtils::printStatsHeader();
    }

    size_t block_size = 16*1024;

    std::cout << "Start processing the batch" << std::endl;
    ret = processBatchSizes(*worker_ptr, iov, block_size);
    if (0 != ret) {
        return EXIT_FAILURE;
    }
    std::cout << "End processing the batch" << std::endl;

    ret = worker_ptr->synchronize(); // Make sure environment is not used anymore
    if (0 != ret) {
        return EXIT_FAILURE;
    }

    gflags::ShutDownCommandLineFlags();

    return worker_ptr->signaled() ? EXIT_FAILURE : EXIT_SUCCESS;
}
