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

#ifndef __NIXL_WORKER_H
#define __NIXL_WORKER_H

#include "config.h"
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <optional>
#include <memory>
#include <nixl.h>
#include "utils/utils.h"
#include "worker/worker.h"

class xferBenchNixlWorker: public xferBenchWorker {
    private:
        nixlAgent* agent;
        nixlBackendH* backend_engine;
        std::vector<std::vector<xferBenchIOV>> remote_iovs;
    public:
        xferBenchNixlWorker(int *argc, char ***argv);
        ~xferBenchNixlWorker();  // Custom destructor to clean up resources

        // Memory management
        std::vector<xferBenchIOV> allocateMemory(void) override;
        void deallocateMemory(std::vector<xferBenchIOV> &iov_list) override;

        // Communication and synchronization
        int exchangeMetadata() override;
        std::vector<xferBenchIOV> exchangeIOV(const std::vector<xferBenchIOV>
                                                    &local_iov_list) override;
        void poll(size_t block_size) override;
        int synchronizeStart();

        // Data operations
        std::variant<double, int> transfer(size_t block_size,
                                           const std::vector<xferBenchIOV> &local_iov_list,
                                           const std::vector<xferBenchIOV> &remote_iov_list) override;

    private:
        std::optional<xferBenchIOV> initBasicDescDram(size_t buffer_size, int mem_dev_id);
        void cleanupBasicDescDram(xferBenchIOV &basic_desc);
};

#endif // __NIXL_WORKER_H
