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

#include "worker/nixl/nixl_worker.h"
#include <cstring>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include "utils/utils.h"
#include <unistd.h>
#include <utility>
#include <sys/time.h>
#include <utils/serdes/serdes.h>
#include <omp.h>

#define USE_VMM 0
#define ROUND_UP(value, granularity) ((((value) + (granularity) - 1) / (granularity)) * (granularity))

#define CHECK_NIXL_ERROR(result, message)                                         \
    do {                                                                          \
        if (0 != result) {                                                        \
            std::cerr << "NIXL: " << message << " (Error code: " << result        \
                      << ")" << std::endl;                                        \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while(0)

xferBenchNixlWorker::xferBenchNixlWorker(int *argc, char ***argv, std::vector<std::string> devices) : xferBenchWorker(argc, argv) {
    seg_type = DRAM_SEG;

    int rank;
    std::string backend_name;
    nixl_b_params_t backend_params;
    bool enable_pt = xferBenchConfig::enable_pt;
    char hostname[256];
    nixl_mem_list_t mems;
    std::vector<nixl_backend_t> plugins;

    rank = rt->getRank();

    nixlAgentConfig dev_meta(enable_pt);

    agent = new nixlAgent(name, dev_meta);

    agent->getAvailPlugins(plugins);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX)){
        backend_name = xferBenchConfig::backend;
    } else {
        std::cerr << "Unsupported backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->getPluginParams(backend_name, mems, backend_params);

    if (0 == xferBenchConfig::backend.compare(XFERBENCH_BACKEND_UCX)){
        // No need to set device_list if all is specified
        // fallback to backend preference
        if (devices[0] != "all" && devices.size() >= 1) {
            if (isInitiator()) {
                backend_params["device_list"] = devices[rank];
            } else {
                backend_params["device_list"] = devices[rank - xferBenchConfig::num_initiator_dev];
            }
        }

        if (gethostname(hostname, 256)) {
           std::cerr << "Failed to get hostname" << std::endl;
           exit(EXIT_FAILURE);
        }

        std::cout << "Init nixl worker, dev " << (("all" == devices[0]) ? "all" : backend_params["device_list"])
                  << " rank " << rank << ", type " << name << ", hostname "
                  << hostname << std::endl;
    } else {
        std::cerr << "Unsupported backend: " << xferBenchConfig::backend << std::endl;
        exit(EXIT_FAILURE);
    }

    agent->createBackend(backend_name, backend_params, backend_engine);
}

xferBenchNixlWorker::~xferBenchNixlWorker() {
    if (agent) {
        delete agent;
        agent = nullptr;
    }
}

// Convert vector of xferBenchIOV to nixl_reg_dlist_t
static void iovListToNixlRegDlist(const std::vector<xferBenchIOV> &iov_list,
                                 nixl_reg_dlist_t &dlist) {
    nixlBlobDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

// Convert nixl_xfer_dlist_t to vector of xferBenchIOV
static std::vector<xferBenchIOV> nixlXferDlistToIOVList(const nixl_xfer_dlist_t &dlist) {
    std::vector<xferBenchIOV> iov_list;
    for (const auto &desc : dlist) {
        iov_list.emplace_back(desc.addr, desc.len, desc.devId);
    }
    return iov_list;
}

// Convert vector of xferBenchIOV to nixl_xfer_dlist_t
static void iovListToNixlXferDlist(const std::vector<xferBenchIOV> &iov_list,
                                  nixl_xfer_dlist_t &dlist) {
    nixlBasicDesc desc;
    for (const auto &iov : iov_list) {
        desc.addr = iov.addr;
        desc.len = iov.len;
        desc.devId = iov.devId;
        dlist.addDesc(desc);
    }
}

std::optional<xferBenchIOV> xferBenchNixlWorker::initBasicDescDram(size_t buffer_size, int mem_dev_id) {
    void *addr;

    addr = calloc(1, buffer_size);
    if (!addr) {
        std::cerr << "Failed to allocate " << buffer_size << " bytes of DRAM memory" << std::endl;
        return std::nullopt;
    }

    if (isInitiator()) {
        memset(addr, XFERBENCH_INITIATOR_BUFFER_ELEMENT, buffer_size);
    } else if (isTarget()) {
        memset(addr, XFERBENCH_TARGET_BUFFER_ELEMENT, buffer_size);
    }

    // TODO: Does device id need to be set for DRAM?
    return std::optional<xferBenchIOV>(std::in_place, (uintptr_t)addr, buffer_size, mem_dev_id);
}

void xferBenchNixlWorker::cleanupBasicDescDram(xferBenchIOV &iov) {
    free((void *)iov.addr);
}

std::vector<std::vector<xferBenchIOV>> xferBenchNixlWorker::allocateMemory(int num_lists) {
    std::vector<std::vector<xferBenchIOV>> iov_lists;
    size_t i, buffer_size, num_devices = 0;
    nixl_opt_args_t opt_args;

    if (isInitiator()) {
        num_devices = xferBenchConfig::num_initiator_dev;
    } else if (isTarget()) {
        num_devices = xferBenchConfig::num_target_dev;
    }
    buffer_size = xferBenchConfig::total_buffer_size / (num_devices * num_lists);

    opt_args.backends.push_back(backend_engine);

    for (int list_idx = 0; list_idx < num_lists; list_idx++) {
        std::vector<xferBenchIOV> iov_list;
        for (i = 0; i < num_devices; i++) {
            std::optional<xferBenchIOV> basic_desc;

            switch (seg_type) {
            case DRAM_SEG:
                basic_desc = initBasicDescDram(buffer_size, i);
                break;
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }

            if (basic_desc) {
                iov_list.push_back(basic_desc.value());
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->registerMem(desc_list, &opt_args),
                       "registerMem failed");
        iov_lists.push_back(iov_list);
    }

    return iov_lists;
}

void xferBenchNixlWorker::deallocateMemory(std::vector<std::vector<xferBenchIOV>> &iov_lists) {
    nixl_opt_args_t opt_args;

    opt_args.backends.push_back(backend_engine);
    for (auto &iov_list: iov_lists) {
        for (auto &iov: iov_list) {
            switch (seg_type) {
            case DRAM_SEG:
                cleanupBasicDescDram(iov);
                break;
            default:
                std::cerr << "Unsupported mem type: " << seg_type << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        nixl_reg_dlist_t desc_list(seg_type);
        iovListToNixlRegDlist(iov_list, desc_list);
        CHECK_NIXL_ERROR(agent->deregisterMem(desc_list, &opt_args),
                         "deregisterMem failed");
    }

}

int xferBenchNixlWorker::exchangeMetadata() {
    int meta_sz, ret = 0;

    if (isTarget()) {
        std::string local_metadata;
        const char *buffer;
        int destrank;

        agent->getLocalMD(local_metadata);

        buffer = local_metadata.data();
        meta_sz = local_metadata.size();

        if (IS_PAIRWISE_AND_SG()) {
            destrank = rt->getRank() - xferBenchConfig::num_target_dev;
            //XXX: Fix up the rank, depends on processes distributed on hosts
            //assumes placement is adjacent ranks to same node
        } else {
            destrank = 0;
        }
        rt->sendInt(&meta_sz, destrank);
        rt->sendChar((char *)buffer, meta_sz, destrank);
    } else if (isInitiator()) {
        char * buffer;
        std::string remote_agent;
        int srcrank;

        if (IS_PAIRWISE_AND_SG()) {
            srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
            //XXX: Fix up the rank, depends on processes distributed on hosts
            //assumes placement is adjacent ranks to same node
        } else {
            srcrank = 1;
        }
        rt->recvInt(&meta_sz, srcrank);
        buffer = (char *)calloc(meta_sz, sizeof(*buffer));
        rt->recvChar((char *)buffer, meta_sz, srcrank);

        std::string remote_metadata(buffer, meta_sz);
        agent->loadRemoteMD(remote_metadata, remote_agent);
        if("" == remote_agent) {
            std::cerr << "NIXL: loadMetadata failed" << std::endl;
        }
        free(buffer);
    }
    return ret;
}

std::vector<std::vector<xferBenchIOV>>
xferBenchNixlWorker::exchangeIOV(const std::vector<std::vector<xferBenchIOV>> &local_iovs) {
    std::vector<std::vector<xferBenchIOV>> res;
    int desc_str_sz;

    for (const auto &local_iov: local_iovs) {
        nixlSerDes ser_des;
        nixl_xfer_dlist_t local_desc(seg_type);

        iovListToNixlXferDlist(local_iov, local_desc);

        if (isTarget()) {
            const char *buffer;
            int destrank;

            local_desc.serialize(&ser_des);
            std::string desc_str = ser_des.exportStr();
            buffer = desc_str.data();
            desc_str_sz = desc_str.size();

            if (IS_PAIRWISE_AND_SG()) {
                destrank = rt->getRank() - xferBenchConfig::num_target_dev;
                //XXX: Fix up the rank, depends on processes distributed on hosts
                //assumes placement is adjacent ranks to same node
            } else {
                destrank = 0;
            }
            rt->sendInt(&desc_str_sz, destrank);
            rt->sendChar((char *)buffer, desc_str_sz, destrank);
        } else if (isInitiator()) {
            char *buffer;
            int srcrank;

            if (IS_PAIRWISE_AND_SG()) {
                srcrank = rt->getRank() + xferBenchConfig::num_initiator_dev;
                //XXX: Fix up the rank, depends on processes distributed on hosts
                //assumes placement is adjacent ranks to same node
            } else {
                srcrank = 1;
            }
            rt->recvInt(&desc_str_sz, srcrank);
            buffer = (char *)calloc(desc_str_sz, sizeof(*buffer));
            rt->recvChar((char *)buffer, desc_str_sz, srcrank);

            std::string desc_str(buffer, desc_str_sz);
            ser_des.importStr(desc_str);

            nixl_xfer_dlist_t remote_desc(&ser_des);
            res.emplace_back(nixlXferDlistToIOVList(remote_desc));
        }
    }
    // Ensure all processes have completed the exchange with a barrier/sync
    synchronize();
    return res;
}

static int execTransfer(nixlAgent *agent,
                        const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                        const std::vector<std::vector<xferBenchIOV>> &remote_iovs,
                        const nixl_xfer_op_t op,
                        const int num_threads)
{
    int ret = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        const auto &local_iov = local_iovs[tid];
        const auto &remote_iov = remote_iovs[tid];

        // TODO: fetch local_desc and remote_desc directly from config
        nixl_xfer_dlist_t local_desc(DRAM_SEG);
        nixl_xfer_dlist_t remote_desc(DRAM_SEG);

        iovListToNixlXferDlist(local_iov, local_desc);
        iovListToNixlXferDlist(remote_iov, remote_desc);

        nixl_opt_args_t params;
        nixl_b_params_t b_params;
        bool error = false;
        nixlXferReqH *req;
        nixl_status_t rc;
        std::string target;

        params.notifMsg = "0xBEEF";
        params.hasNotif = true;
        target = "target";

        CHECK_NIXL_ERROR(agent->createXferReq(op, local_desc, remote_desc, target,
                                            req, &params), "createTransferReq failed");

        rc = agent->postXferReq(req);
        if (NIXL_ERR_BACKEND == rc) {
            std::cout << "NIXL postRequest failed" << std::endl;
            error = true;
        } else {
            do {
                /* XXX agent isn't const because the getXferStatus() is not const  */
                rc = agent->getXferStatus(req);
                if (NIXL_ERR_BACKEND == rc) {
                    std::cout << "NIXL getStatus failed" << std::endl;
                    error = true;
                    break;
                }
            } while (NIXL_SUCCESS != rc);
        }

        agent->releaseXferReq(req);
        if (error) {
            std::cout << "NIXL releaseXferReq failed" << std::endl;
            ret = -1;
        }
    }

    return ret;
}

std::variant<double, int> xferBenchNixlWorker::transfer(size_t block_size,
                                               const std::vector<std::vector<xferBenchIOV>> &local_iovs,
                                               const std::vector<std::vector<xferBenchIOV>> &remote_iovs) {
    struct timeval t_start, t_end;
    double total_duration = 0.0;
    int ret = 0;
    nixl_xfer_op_t xfer_op = XFERBENCH_OP_READ == xferBenchConfig::op_type ? NIXL_READ : NIXL_WRITE;
    // int completion_flag = 1;

    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, xferBenchConfig::num_threads);
    if (ret < 0) {
        return std::variant<double, int>(ret);
    }
    
    // Synchronize to ensure all processes have completed the warmup (iter and polling)
    synchronize();

    gettimeofday(&t_start, nullptr);

    ret = execTransfer(agent, local_iovs, remote_iovs, xfer_op, xferBenchConfig::num_threads);

    gettimeofday(&t_end, nullptr);
    total_duration += (((t_end.tv_sec - t_start.tv_sec) * 1e6) +
                       (t_end.tv_usec - t_start.tv_usec)); // In us

    return ret < 0 ? std::variant<double, int>(ret) : std::variant<double, int>(total_duration);
}

void xferBenchNixlWorker::poll(size_t block_size) {
    nixl_notifs_t notifs;
    int skip = 1;
    int total_iter = 2;

    /* Ensure warmup is done*/
    while (skip != int(notifs["initiator"].size())) {
        agent->getNotifs(notifs);
    }
    synchronize();

    /* Polling for actual iterations*/
    while (total_iter != int(notifs["initiator"].size())) {
        agent->getNotifs(notifs);
    }
}

int xferBenchNixlWorker::synchronizeStart() {
    if (IS_PAIRWISE_AND_SG()) {
    	std::cout << "Waiting for all processes to start... (expecting "
    	          << rt->getSize() << " total: "
		  << xferBenchConfig::num_initiator_dev << " initiators and "
    	          << xferBenchConfig::num_target_dev << " targets)" << std::endl;
    } else {
    	std::cout << "Waiting for all processes to start... (expecting "
    	          << rt->getSize() << " total" << std::endl;
    }
    if (rt) {
        int ret = rt->barrier("start_barrier");
        if (ret != 0) {
            std::cerr << "Failed to synchronize at start barrier" << std::endl;
            return -1;
        }
        std::cout << "All processes are ready to proceed" << std::endl;
        return 0;
    }
    return -1;
}
