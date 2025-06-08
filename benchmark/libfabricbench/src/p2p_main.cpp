#include <iostream>
#include <nixl.h>
#include <memory>
#include <string>
#include <thread>
#include <chrono>
#include <optional>
#include <etcd/Client.hpp>
#include <utils/serdes/serdes.h>

class EtcdHelper {
private:
    std::unique_ptr<etcd::Client> client;
    std::string prefix;
    int rank;

public:
    EtcdHelper(const std::string& endpoints, int rank) 
        : client(std::make_unique<etcd::Client>(endpoints)), 
          prefix("/nixl_p2p/"), 
          rank(rank) {}

    bool exchangeDesc(const nixl_reg_dlist_t& local_desc, nixl_reg_dlist_t& remote_desc) {
        try {
            if (rank == 0) { // initiator
                // Wait for target to publish its descriptor
                std::string target_key = prefix + "target_desc";
                auto resp = waitForKey(target_key);
                if (!resp.has_value()) return false;

                // Deserialize target's descriptor
                nixlSerDes ser_des;
                ser_des.importStr(resp.value());
                remote_desc = nixl_reg_dlist_t(&ser_des);

                // Publish initiator's descriptor
                nixlSerDes local_ser_des;
                local_desc.serialize(&local_ser_des);
                client->put(prefix + "init_desc", local_ser_des.exportStr()).get();
            } else { // target
                // Publish target's descriptor
                nixlSerDes local_ser_des;
                local_desc.serialize(&local_ser_des);
                client->put(prefix + "target_desc", local_ser_des.exportStr()).get();

                // Wait for initiator's descriptor
                std::string init_key = prefix + "init_desc";
                auto resp = waitForKey(init_key);
                if (!resp.has_value()) return false;

                // Deserialize initiator's descriptor
                nixlSerDes ser_des;
                ser_des.importStr(resp.value());
                remote_desc = nixl_reg_dlist_t(&ser_des);
            }
            return true;
        } catch (const std::exception& e) {
            std::cerr << "ETCD exchange error: " << e.what() << std::endl;
            return false;
        }
    }

    bool barrier(const std::string& barrier_name) {
        try {
            std::string barrier_key = prefix + "barrier/" + barrier_name;
            
            // Signal arrival at barrier
            client->put(barrier_key + "/" + std::to_string(rank), "1").get();
            
            // Wait for both processes
            int retries = 0;
            while (retries < 30) {
                auto resp = client->ls(barrier_key).get();
                if (resp.keys().size() == 2) { // both processes arrived
                    if (rank == 0) { // cleanup by initiator
                        client->rmdir(barrier_key, true).get();
                    }
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::seconds(1));
                retries++;
            }
            return false;
        } catch (const std::exception& e) {
            std::cerr << "Barrier error: " << e.what() << std::endl;
            return false;
        }
    }

private:
    std::optional<std::string> waitForKey(const std::string& key) {
        int retries = 0;
        while (retries < 30) {
            auto resp = client->get(key).get();
            if (resp.error_code() == 0) {
                return resp.value().as_string();
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            retries++;
        }
        return std::nullopt;
    }
};

class SimpleWorker {
private:
    std::string name;
    bool is_initiator;
    nixlAgent* agent;
    nixlBackendH* backend_engine;
    static constexpr size_t BLOCK_SIZE = 4096;  // 4KB block
    std::unique_ptr<EtcdHelper> etcd;

public:
    SimpleWorker(int rank, const std::string& etcd_endpoints) 
        : name(rank == 0 ? "initiator" : "target"),
          is_initiator(rank == 0),
          etcd(std::make_unique<EtcdHelper>(etcd_endpoints, rank)) {
        
        // Initialize NIXL agent
        nixlAgentConfig dev_meta(false);  // disable progress thread
        agent = new nixlAgent(name, dev_meta);

        // Setup UCX backend
        nixl_b_params_t backend_params;
        nixl_mem_list_t mems;
        std::vector<nixl_backend_t> plugins;
        
        agent->getAvailPlugins(plugins);
        agent->getPluginParams("UCX", mems, backend_params);
        agent->createBackend("UCX", backend_params, backend_engine);
    }

    ~SimpleWorker() {
        if (agent) {
            delete agent;
        }
    }

    int transfer() {
        // Synchronize start
        if (!etcd->barrier("start")) {
            std::cerr << "Failed at start barrier" << std::endl;
            return -1;
        }

        if (is_initiator) {
            return initiator_transfer();
        } else {
            return target_transfer();
        }
    }

private:
    private:
    private:
    int initiator_transfer() {
        void* send_buffer = calloc(1, BLOCK_SIZE);
        if (!send_buffer) {
            std::cerr << "Failed to allocate send buffer" << std::endl;
            return -1;
        }
        memset(send_buffer, 0xbb, BLOCK_SIZE);

        // Create registration descriptors
        nixl_reg_dlist_t reg_local_desc(DRAM_SEG);
        nixl_reg_dlist_t reg_remote_desc(DRAM_SEG);
        
        nixlBlobDesc reg_buffer;
        reg_buffer.addr = reinterpret_cast<uintptr_t>(send_buffer);
        reg_buffer.len = BLOCK_SIZE;
        reg_buffer.devId = 0;
        reg_local_desc.addDesc(reg_buffer);

        // Register memory
        nixl_opt_args_t opt_args;
        opt_args.backends.push_back(backend_engine);
        if (agent->registerMem(reg_local_desc, &opt_args)) {
            std::cerr << "Failed to register memory" << std::endl;
            free(send_buffer);
            return -1;
        }

        // Exchange descriptors via ETCD
        if (!etcd->exchangeDesc(reg_local_desc, reg_remote_desc)) {
            std::cerr << "Failed to exchange descriptors" << std::endl;
            free(send_buffer);
            return -1;
        }

        // Create transfer descriptors
        nixl_xfer_dlist_t xfer_local_desc(DRAM_SEG);
        nixl_xfer_dlist_t xfer_remote_desc(DRAM_SEG);

        // Add local transfer descriptor
        nixlBasicDesc xfer_buffer;
        xfer_buffer.addr = reinterpret_cast<uintptr_t>(send_buffer);
        xfer_buffer.len = BLOCK_SIZE;
        xfer_buffer.devId = 0;
        xfer_local_desc.addDesc(xfer_buffer);

        // Convert and add remote descriptor
        for (const auto& reg_desc : reg_remote_desc) {
            nixlBasicDesc remote_xfer_buffer;
            remote_xfer_buffer.addr = reg_desc.addr;
            remote_xfer_buffer.len = reg_desc.len;
            remote_xfer_buffer.devId = reg_desc.devId;
            xfer_remote_desc.addDesc(remote_xfer_buffer);
        }

        // Create and post transfer request
        nixlXferReqH* req;
        nixl_opt_args_t params;
        params.notifMsg = "transfer";
        params.hasNotif = true;

        if (agent->createXferReq(NIXL_WRITE, xfer_local_desc, xfer_remote_desc, "target", req, &params)) {
            std::cerr << "Failed to create transfer request" << std::endl;
            free(send_buffer);
            return -1;
        }

        nixl_status_t rc = agent->postXferReq(req);
        if (rc != NIXL_SUCCESS) {
            std::cerr << "Failed to post transfer request" << std::endl;
            free(send_buffer);
            return -1;
        }

        // Wait for completion
        do {
            rc = agent->getXferStatus(req);
        } while (rc == NIXL_IN_PROG);

        agent->releaseXferReq(req);
        agent->deregisterMem(reg_local_desc, &opt_args);
        free(send_buffer);
        
        return (rc == NIXL_SUCCESS) ? 0 : -1;
    }



    int target_transfer() {
        void* recv_buffer = calloc(1, BLOCK_SIZE);
        if (!recv_buffer) {
            std::cerr << "Failed to allocate receive buffer" << std::endl;
            return -1;
        }

        // Create registration descriptors
        nixl_reg_dlist_t reg_local_desc(DRAM_SEG);
        nixl_reg_dlist_t reg_remote_desc(DRAM_SEG);
        
        nixlBlobDesc reg_buffer;
        reg_buffer.addr = reinterpret_cast<uintptr_t>(recv_buffer);
        reg_buffer.len = BLOCK_SIZE;
        reg_buffer.devId = 0;
        reg_local_desc.addDesc(reg_buffer);

        // Register memory
        nixl_opt_args_t opt_args;
        opt_args.backends.push_back(backend_engine);
        if (agent->registerMem(reg_local_desc, &opt_args)) {
            std::cerr << "Failed to register memory" << std::endl;
            free(recv_buffer);
            return -1;
        }

        // Exchange descriptors via ETCD
        if (!etcd->exchangeDesc(reg_local_desc, reg_remote_desc)) {
            std::cerr << "Failed to exchange descriptors" << std::endl;
            free(recv_buffer);
            return -1;
        }

        // Wait for transfer completion
        nixl_notifs_t notifs;
        while (notifs["initiator"].empty()) {
            agent->getNotifs(notifs);
        }

        // Verify received data
        uint8_t* data = static_cast<uint8_t*>(recv_buffer);
        bool data_valid = true;
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
            if (data[i] != 0xbb) {
                std::cerr << "Data verification failed at offset " << i << std::endl;
                data_valid = false;
                break;
            }
        }

        agent->deregisterMem(reg_local_desc, &opt_args);
        free(recv_buffer);
        
        return data_valid ? 0 : -1;
    }

};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <rank>" << std::endl;
        return 1;
    }

    int rank = std::stoi(argv[1]);
    std::string etcd_endpoints = "http://10.0.59.59:2379";

    try {
        SimpleWorker worker(rank, etcd_endpoints);
        int result = worker.transfer();
        std::cout << (rank == 0 ? "Initiator" : "Target") 
                  << " completed with result: " << result << std::endl;
        return result;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
