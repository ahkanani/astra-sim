/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/Workload.hh"

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/IntData.hh"
#include "astra-sim/system/MemEventHandlerData.hh"
#include "astra-sim/system/RecvPacketEventHandlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/system/AstraMemoryAPI.hh"
#include <json/json.hpp>

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <map>
#include <limits>
#include <stdexcept>

using namespace std;
using namespace AstraSim;
using namespace Chakra;
using json = nlohmann::json;
using AstraSim::MemoryLocationType; 

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

namespace {
// Variable-size expert exchanges use one collective barrier for GPU and CPU/PIM
// domains alike. Peer messages travel through the normal network API, including
// the configured scale-up links and shared full-duplex scale-out NIC queues.
// Legacy scalar-size collectives continue through the existing ASTRA plans.
struct ExpertExchangeParticipant {
    Workload* workload;
    uint64_t node_id;
    Tick issued;
    std::vector<uint64_t> sends, receives;
};

struct ExpertExchangeState {
    std::vector<int> members;
    std::map<int, ExpertExchangeParticipant> participants;
    size_t remaining = 0;
    int tag;
};

using ExpertExchangeKey = std::pair<int, uint64_t>;
std::map<ExpertExchangeKey, std::shared_ptr<ExpertExchangeState>> expert_exchanges;
int next_expert_exchange_tag = 0;

void finish_expert_exchange(const std::shared_ptr<ExpertExchangeState>& state) {
    for (const auto& entry : state->participants) {
        const auto& participant = entry.second;
        auto* workload = participant.workload;
        workload->hw_resource->tics_gpu_comms += Sys::boostedTick() - participant.issued;
        auto* completion = new WorkloadLayerHandlerData;
        completion->node_id = participant.node_id;
        workload->sys->register_event(workload, EventType::General, completion, 0);
    }
}

void expert_exchange_received(void* argument) {
    auto* holder = static_cast<std::shared_ptr<ExpertExchangeState>*>(argument);
    auto state = *holder;
    delete holder;
    if (--state->remaining == 0) finish_expert_exchange(state);
}

void issue_expert_exchange(Workload* workload,
                          const std::shared_ptr<Chakra::ETFeederNode>& node,
                          CommunicatorGroup* group) {
    if (group == nullptr || node->comm_type() != ChakraCollectiveCommType::ALL_TO_ALL)
        throw std::runtime_error("Expert exchange requires an ALL_TO_ALL communicator");
    const auto& send_attr = node->get_other_attr("replay_send_counts");
    const auto& recv_attr = node->get_other_attr("replay_recv_counts");
    if (!send_attr.has_uint64_list() || !recv_attr.has_uint64_list())
        throw std::runtime_error("Expert exchange requires uint64 count vectors");
    const auto& sends = send_attr.uint64_list().values();
    const auto& receives = recv_attr.uint64_list().values();
    if (sends.size() != group->involved_NPUs.size() || receives.size() != sends.size())
        throw std::runtime_error("Expert exchange counts do not match communicator");
    const ExpertExchangeKey key(node->get_other_attr("comm_group_id").int32_val(),
                                node->get_other_attr("replay_collective_id").uint64_val());
    auto& slot = expert_exchanges[key];
    if (!slot) {
        slot = std::make_shared<ExpertExchangeState>();
        slot->members = group->involved_NPUs;
        if (next_expert_exchange_tag == 500000000)
            throw std::runtime_error("Expert exchange message tag space exhausted");
        slot->tag = next_expert_exchange_tag++;
    }
    auto state = slot;
    if (state->members != group->involved_NPUs || state->participants.count(workload->sys->id))
        throw std::runtime_error("Inconsistent or duplicate expert collective participant");
    state->participants.emplace(workload->sys->id, ExpertExchangeParticipant{
        workload, node->id(), Sys::boostedTick(),
        std::vector<uint64_t>(sends.begin(), sends.end()),
        std::vector<uint64_t>(receives.begin(), receives.end())});
    if (state->participants.size() != state->members.size()) return;

    // Validate the complete matrix before reserving any network work. Self data
    // is already resident and must not be charged as a network transfer.
    for (size_t i = 0; i < state->members.size(); ++i) {
        const auto& source = state->participants.at(state->members[i]);
        uint64_t total = 0;
        for (size_t j = 0; j < state->members.size(); ++j) {
            const auto bytes = source.sends[j];
            const auto& target = state->participants.at(state->members[j]);
            if (bytes != target.receives[i] || (i == j && bytes != 0))
                throw std::runtime_error("Expert exchange send/receive matrix mismatch");
            if (bytes > std::numeric_limits<uint64_t>::max() - total)
                throw std::runtime_error("Expert exchange byte count overflow");
            total += bytes;
            if (bytes) ++state->remaining;
        }
        if (total != source.workload->et_feeder->lookupNode(source.node_id)->comm_size())
            throw std::runtime_error("Expert exchange comm_size disagrees with count vector");
    }
    expert_exchanges.erase(key);
    if (state->remaining == 0) {
        finish_expert_exchange(state);
        return;
    }
    // Direct all-to-all-v: packed, nonempty peer exchanges only. This is not a
    // dense padded ring; GPU and PIM peers use exactly the same implementation.
    for (size_t i = 0; i < state->members.size(); ++i) {
        const auto& source = state->participants.at(state->members[i]);
        for (size_t j = 0; j < state->members.size(); ++j) {
            const auto bytes = source.sends[j];
            if (!bytes) continue;
            const auto& target = state->participants.at(state->members[j]);
            sim_request request{};
            request.srcRank = state->members[i];
            request.dstRank = state->members[j];
            request.reqType = UINT8;
            target.workload->sys->front_end_sim_recv(0, Sys::dummy_data, bytes, UINT8,
                state->members[i], state->tag, &request, Sys::FrontEndSendRecvType::NATIVE,
                expert_exchange_received, new std::shared_ptr<ExpertExchangeState>(state));
            source.workload->sys->front_end_sim_send(0, Sys::dummy_data, bytes, UINT8,
                state->members[j], state->tag, &request, Sys::FrontEndSendRecvType::NATIVE,
                [](void*) {}, nullptr);
        }
    }
}
}  // namespace

Workload::Workload(Sys* sys, string et_filename, string comm_group_filename) {
    string workload_filename = et_filename + "." + to_string(sys->id) + ".et";
    // Check if workload filename exists
    if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
            error_msg =
                "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
            error_msg = "workload file: " + workload_filename +
                        " exists but is not readable";
        } else {
            error_msg =
                "Unknown workload file: " + workload_filename + " access error";
        }
        LoggerFactory::get_logger("workload")->critical(error_msg);
        exit(EXIT_FAILURE);
    }
    this->et_feeder = new ETFeeder(workload_filename);
    this->comm_group = nullptr;
    this->hw_resource = new HardwareResource(1);
    this->sys = sys;
    initialize_comm_group(comm_group_filename);
    this->is_finished = false;
    this->iteration = 0;
    this->filename = et_filename;
    this->is_sleep = false;
}

Workload::~Workload() {
    for (auto& entry : comm_groups) delete entry.second;
    if (this->comm_group != nullptr) {
        delete this->comm_group;
    }
    if (this->et_feeder != nullptr) {
        delete this->et_feeder;
    }
    if (this->hw_resource != nullptr) {
        delete this->hw_resource;
    }
}

void Workload::initialize_comm_group(string comm_group_filename) {
    // communicator group input file is not given
    if (comm_group_filename.find("empty") != std::string::npos) {
        comm_group = nullptr;
        return;
    }

    ifstream inFile;
    json j;
    inFile.open(comm_group_filename);
    inFile >> j;

    if (j.contains("groups")) {
        for (auto it = j["groups"].begin(); it != j["groups"].end(); ++it) {
            std::vector<int> members = it.value().get<std::vector<int>>();
            if (std::find(members.begin(), members.end(), sys->id) != members.end()) {
                int group_id = std::stoi(it.key());
                comm_groups[group_id] = new CommunicatorGroup(group_id, members, sys);
            }
        }
        return;
    }

    for (json::iterator it = j.begin(); it != j.end(); ++it) {
        bool in_comm_group = false;

        for (auto id : it.value()) {
            if (id == sys->id) {
                in_comm_group = true;
            }
        }

        if (in_comm_group) {
            std::vector<int> involved_NPUs;
            for (auto id : it.value()) {
                involved_NPUs.push_back(id);
            }
            comm_group = new CommunicatorGroup(1, involved_NPUs, sys);
            // Note: All NPUs should create comm group with identical ids if
            // they want to communicate with each other
        }
    }
}

void Workload::issue_dep_free_nodes() {
    std::queue<shared_ptr<Chakra::ETFeederNode>> push_back_queue;
    shared_ptr<Chakra::ETFeederNode> node = et_feeder->getNextIssuableNode();
    while (node != nullptr) {
        if (hw_resource->is_available(node)) {
            issue(node);
        } else {
            push_back_queue.push(node);
        }
        node = et_feeder->getNextIssuableNode();
    }

    while (!push_back_queue.empty()) {
        shared_ptr<Chakra::ETFeederNode> node = push_back_queue.front();
        et_feeder->pushBackIssuableNode(node->id());
        push_back_queue.pop();
    }
}

void Workload::issue(shared_ptr<Chakra::ETFeederNode> node) {
    auto logger = LoggerFactory::get_logger("workload");
    if (sys->replay_only) {
        hw_resource->occupy(node);
        issue_replay(node);
    } else {
        if ((node->type() == ChakraNodeType::MEM_LOAD_NODE) ||
            (node->type() == ChakraNodeType::MEM_STORE_NODE) ||
            (node->type() == ChakraNodeType::PIM_COMP_NODE)) {
            if (sys->trace_enabled) {
                logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                              "node->name={}, node->type={}",
                              sys->id, Sys::boostedTick(), node->id(),
                              node->name(),
                              static_cast<uint64_t>(node->type()));
            }
            issue_mem(node);
        } else if (node->is_cpu_op() ||
                   (!node->is_cpu_op() &&
                    node->type() == ChakraNodeType::COMP_NODE)) {
            if ((node->runtime() == 0) && (node->num_ops() == 0)) {
                skip_invalid(node);
            } else {
                if (sys->trace_enabled) {
                    logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                                  "node->name={}, node->type={}",
                                  sys->id, Sys::boostedTick(), node->id(),
                                  node->name(),
                                  static_cast<uint64_t>(node->type()));
                }
                issue_comp(node);
            }
        } else if (!node->is_cpu_op() &&
                   (node->type() == ChakraNodeType::COMM_COLL_NODE ||
                    (node->type() == ChakraNodeType::COMM_SEND_NODE) ||
                    (node->type() == ChakraNodeType::COMM_RECV_NODE))) {
            if (sys->trace_enabled) {
                if (sys->trace_enabled) {
                    logger->debug("issue,sys->id={}, tick={}, node->id={}, "
                                  "node->name={}, node->type={}",
                                  sys->id, Sys::boostedTick(), node->id(),
                                  node->name(),
                                  static_cast<uint64_t>(node->type()));
                }
            }
            issue_comm(node);
        } else if (node->type() == ChakraNodeType::INVALID_NODE) {
            skip_invalid(node);
        }
    }
}

void Workload::issue_replay(shared_ptr<Chakra::ETFeederNode> node) {
    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node->id();
    uint64_t runtime = 1ul;
    if (node->runtime() != 0ul) {
        // chakra runtimes are in microseconds and we should convert it into
        // nanoseconds
        // -> changed chakra to return nanoseconds
        runtime = node->runtime(); // * 1000;
    }
    if (node->is_cpu_op()) {
        hw_resource->tics_cpu_ops += runtime;
    } else {
        hw_resource->tics_gpu_ops += runtime;
    }
    sys->register_event(this, EventType::General, wlhd, runtime);
}

/* 
    Now the node only stores one tensor_size for MEM_LOAD/STORE
    Can get tensor size using
    - node->tensor_size()
    Loading/storing of the input/output is hardcoded in chakra
    But, still can get input types from the node name
    - node->name().find("INPUT") != string::npos
    - node->name().find("OUTPUT") != string::npos
    - node->name().find("WEIGHT") != string::npos
    
    For the tensor location use tensor_loc()
    - node->tensor_loc() 
    INVALID_MEMORY = 0, LOCAL_MEMORY = 1, REMOTE_MEMORY = 2, CXL_MEMORY = 3 STORAGE_MEMORY = 4
*/
void Workload::issue_mem(shared_ptr<Chakra::ETFeederNode> node) {
    hw_resource->occupy(node);

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->sys_id = sys->id;
    wlhd->workload = this;
    wlhd->node_id = node->id();
    wlhd->device_id = node->tensor_device();
    if (node->type() == ChakraNodeType::PIM_COMP_NODE) { // pim implementation
        wlhd->pim_enabled = true;
        wlhd->pim_channel_id = node->tensor_channel();
        wlhd->pim_runtime = node->runtime();
    }
    switch (static_cast<MemoryLocationType>(node->tensor_loc())) {
        case MemoryLocationType::LOCAL_MEMORY:
            // local memory access
            sys->local_mem->issue(node->tensor_size(), wlhd);
            break;
        case MemoryLocationType::REMOTE_MEMORY:
            // remote memory access
            sys->remote_mem->issue(node->tensor_size(), wlhd);
            break;
        case MemoryLocationType::CXL_MEMORY:
            // CXL memory access
            sys->cxl_mem->issue(node->tensor_size(), wlhd);
            break;
        case MemoryLocationType::STORAGE_MEMORY:
            // storage memory access
            sys->storage_mem->issue(node->tensor_size(), wlhd);
            break;
        case MemoryLocationType::INVALID_MEMORY:
        default:
            // invalid memory access
            LoggerFactory::get_logger("workload")->critical("Invalid memory type");
            exit(EXIT_FAILURE);
    }
}

void Workload::issue_comp(shared_ptr<Chakra::ETFeederNode> node) {
    hw_resource->occupy(node);

    if (sys->roofline_enabled) {
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->node_id = node->id();

        double operational_intensity = static_cast<double>(node->num_ops()) /
                                       static_cast<double>(node->tensor_size());
        double perf = sys->roofline->get_perf(operational_intensity);
        double elapsed_time =
            static_cast<double>(node->num_ops()) / perf;  // sec
        uint64_t runtime =
            static_cast<uint64_t>(elapsed_time * 1e9);  // sec -> ns
        if (node->is_cpu_op()) {
            hw_resource->tics_cpu_ops += runtime;
        } else {
            hw_resource->tics_gpu_ops += runtime;
        }
        sys->register_event(this, EventType::General, wlhd, runtime);
    } else {
        // advance this node forward the recorded "replayed" time specificed in
        // the ET.
        issue_replay(node);
    }
}

void Workload::issue_comm(shared_ptr<Chakra::ETFeederNode> node) {
    CommunicatorGroup* selected_group = comm_group;
    if (node->has_other_attr("comm_group_id")) {
        int group_id = node->get_other_attr("comm_group_id").int32_val();
        auto group = comm_groups.find(group_id);
        if (group == comm_groups.end()) throw std::runtime_error("Unknown collective group on this GPU");
        selected_group = group->second;
    }
    hw_resource->occupy(node);

    if (node->has_other_attr("replay_alltoallv") &&
        node->get_other_attr("replay_alltoallv").bool_val()) {
        issue_expert_exchange(this, node, selected_group);
        return;
    }

    vector<bool> involved_dim;

    if (node->has_other_attr("involved_dim")) {
        const ChakraProtoMsg::AttributeProto& attr =
            node->get_other_attr("involved_dim");

        // Ensure the attribute is of type bool_list before accessing
        if (attr.has_bool_list()) {
            const ChakraProtoMsg::BoolList& bool_list = attr.bool_list();

            // Traverse bool_list and add values to involved_dim
            for (int i = 0; i < bool_list.values_size(); ++i) {
                involved_dim.push_back(bool_list.values(i));
            }
        } else {
            cerr << "Expected bool_list in involved_dim but found another type."
                 << endl;
            exit(EXIT_FAILURE);
        }
    } else {
        // involved_dim does not exist in ETFeeder.
        // Assume involved_dim = [1,1,1,1,1] which we could simulate 5-Dimension.
	// Could use Process Group to build involved_dim later. 
	// Once process group is implemented, you should get
        // that with node->pg_name()
	
	for(int i = 0; i < 4; i++)
            involved_dim.push_back(true);
    }

    if (!node->is_cpu_op() &&
        (node->type() == ChakraNodeType::COMM_COLL_NODE)) {
        if (node->comm_type() == ChakraCollectiveCommType::ALL_REDUCE) {
            DataSet* fp =
                sys->generate_all_reduce(node->comm_size(), involved_dim,
                                         selected_group, node->comm_priority());
            collective_comm_node_id_map[fp->my_id] = node->id();
            collective_comm_wrapper_map[fp->my_id] = fp;
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

        } else if (node->comm_type() == ChakraCollectiveCommType::ALL_TO_ALL) {
            DataSet* fp =
                sys->generate_all_to_all(node->comm_size(), involved_dim,
                                         selected_group, node->comm_priority());
            collective_comm_node_id_map[fp->my_id] = node->id();
            collective_comm_wrapper_map[fp->my_id] = fp;
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

        } else if (node->comm_type() == ChakraCollectiveCommType::ALL_GATHER) {
            DataSet* fp =
                sys->generate_all_gather(node->comm_size(), involved_dim,
                                         selected_group, node->comm_priority());
            collective_comm_node_id_map[fp->my_id] = node->id();
            collective_comm_wrapper_map[fp->my_id] = fp;
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

        } else if (node->comm_type() ==
                   ChakraCollectiveCommType::REDUCE_SCATTER) {
            DataSet* fp =
                sys->generate_reduce_scatter(node->comm_size(), involved_dim,
                                             selected_group, node->comm_priority());
            collective_comm_node_id_map[fp->my_id] = node->id();
            collective_comm_wrapper_map[fp->my_id] = fp;
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);

        } else if (node->comm_type() == ChakraCollectiveCommType::BROADCAST) {
            // broadcast colelctive has not been implemented in ASTRA-SIM yet.
            // So, we just use its real system mesurements
            uint64_t runtime = 1ul;
            if (node->runtime() != 0ul) {
                // chakra runtimes are in microseconds and we should convert it
                // into nanoseconds
                // -> changed chakra to return nanoseconds
                runtime = node->runtime(); // * 1000;
            }
            DataSet* fp = new DataSet(1);
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
            collective_comm_node_id_map[fp->my_id] = node->id();
            collective_comm_wrapper_map[fp->my_id] = fp;
            sys->register_event(fp, EventType::General, nullptr,
                                // chakra runtimes are in microseconds and we
                                // should convert it into nanoseconds
                                runtime);
            fp->set_notifier(this, EventType::CollectiveCommunicationFinished);
        }
    } else if (node->type() == ChakraNodeType::COMM_SEND_NODE) {
        sim_request snd_req;
        snd_req.srcRank = node->comm_src();
        snd_req.dstRank = node->comm_dst();
        snd_req.reqType = UINT8;
        SendPacketEventHandlerData* sehd = new SendPacketEventHandlerData;
        sehd->callable = this;
        sehd->wlhd = new WorkloadLayerHandlerData;
        sehd->wlhd->node_id = node->id();
        sehd->event = EventType::PacketSent;
        sys->front_end_sim_send(0, Sys::dummy_data, node->comm_size(), UINT8,
                                node->comm_dst(), node->comm_tag(), &snd_req,
                                Sys::FrontEndSendRecvType::NATIVE,
                                &Sys::handleEvent, sehd);
    } else if (node->type() == ChakraNodeType::COMM_RECV_NODE) {
        sim_request rcv_req;
        RecvPacketEventHandlerData* rcehd = new RecvPacketEventHandlerData;
        rcehd->wlhd = new WorkloadLayerHandlerData;
        rcehd->wlhd->node_id = node->id();
        rcehd->workload = this;
        rcehd->event = EventType::PacketReceived;
        sys->front_end_sim_recv(0, Sys::dummy_data, node->comm_size(), UINT8,
                                node->comm_src(), node->comm_tag(), &rcv_req,
                                Sys::FrontEndSendRecvType::NATIVE,
                                &Sys::handleEvent, rcehd);
    } else {
        LoggerFactory::get_logger("workload")
            ->critical("Unknown communication node type");
        exit(EXIT_FAILURE);
    }
}

void Workload::skip_invalid(shared_ptr<Chakra::ETFeederNode> node) {
    et_feeder->freeChildrenNodes(node->id());
    et_feeder->removeNode(node->id());
}

void Workload::call(EventType event, CallData* data) {
    if (is_finished) {
        return;
    }

    if (event == EventType::CollectiveCommunicationFinished) {
        IntData* int_data = (IntData*)data;
        hw_resource->tics_gpu_comms += int_data->execution_time;
        uint64_t node_id = collective_comm_node_id_map[int_data->data];
        shared_ptr<Chakra::ETFeederNode> node = et_feeder->lookupNode(node_id);

        if (sys->trace_enabled) {
            LoggerFactory::get_logger("workload")
                ->debug("callback,sys->id={}, tick={}, node->id={}, "
                        "node->name={}, node->type={}",
                        sys->id, Sys::boostedTick(), node->id(), node->name(),
                        static_cast<uint64_t>(node->type()));
        }

        hw_resource->release(node);

        et_feeder->freeChildrenNodes(node_id);

        issue_dep_free_nodes();
      
        // The Dataset class provides statistics that should be used later to dump
        // more statistics in the workload layer
        delete collective_comm_wrapper_map[int_data->data];
        collective_comm_wrapper_map.erase(int_data->data);
        et_feeder->removeNode(node_id);

    } else {
        if (data == nullptr) {
            issue_dep_free_nodes();
        } else {
            WorkloadLayerHandlerData* wlhd = (WorkloadLayerHandlerData*)data;
            shared_ptr<Chakra::ETFeederNode> node =
                et_feeder->lookupNode(wlhd->node_id);

            if (sys->trace_enabled) {
                LoggerFactory::get_logger("workload")
                    ->debug("callback,sys->id={}, tick={}, node->id={}, "
                            "node->name={}, node->type={}",
                            sys->id, Sys::boostedTick(), node->id(),
                            node->name(), static_cast<uint64_t>(node->type()));
            }

            hw_resource->release(node);

            et_feeder->freeChildrenNodes(node->id());

            issue_dep_free_nodes();

            et_feeder->removeNode(wlhd->node_id);
            delete wlhd;
        }
    }

    if (!et_feeder->hasNodesToIssue() &&
        (hw_resource->num_in_flight_cpu_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comp_ops == 0) &&
        (hw_resource->num_in_flight_gpu_comm_ops == 0) ) {
        // report();
        if (!pending_workloads.empty()) {
            string next_workload = pending_workloads.front();
            pending_workloads.pop();
            // there exists new workload, change the ETFeeder
            if (this->et_feeder != nullptr)
                delete this->et_feeder;
            this->et_feeder = new ETFeeder(next_workload);
            iteration++;
            is_finished = false;
            // fire next iteration
            fire();
        } else {
            // wait for the new workload command
            // should be fired by parent controller
            sys->comm_NI->sim_notify_finished();
            is_finished = true;
        }
    }
}

void Workload::add_workload(const std::string& new_filename,
                            const std::vector<Sys*>& systems) {

    // add new workload filename to pending workloads in managed systems
    for (auto* managed_sys : systems) {
        if (managed_sys == nullptr || managed_sys->workload == nullptr) {
            LoggerFactory::get_logger("workload")
                ->critical("Null system or workload while adding workload {}", new_filename);
        }
        string workload_filename = new_filename + "." + to_string(managed_sys->id) + ".et";
        // Check if workload filename exists
        if (access(workload_filename.c_str(), R_OK) < 0) {
            string error_msg;
            if (errno == ENOENT) {
                error_msg = "workload file: " + workload_filename + " does not exist";
            } else if (errno == EACCES) {
                error_msg = "workload file: " + workload_filename + " exists but is not readable";
            } else {
                error_msg = "Unknown workload file: " + workload_filename + " access error";
            }
            LoggerFactory::get_logger("workload")->critical(error_msg);
            return;
        }
        if (managed_sys->workload->is_finished && managed_sys->workload->pending_workloads.empty()) {
            // if the workload is finished, we can directly change the ETFeeder
            if (managed_sys->workload->et_feeder != nullptr)
                delete managed_sys->workload->et_feeder;
            managed_sys->workload->et_feeder = new ETFeeder(workload_filename);
            managed_sys->workload->iteration++;
            managed_sys->workload->is_finished = false;
            // fire next iteration
            managed_sys->workload->fire();
            continue;
        }
        else{
            managed_sys->workload->pending_workloads.push(workload_filename);
        }
    }
    string workload_filename = new_filename + "." + to_string(sys->id) + ".et";
    // cout << workload_filename << endl;
  
    // Check if workload filename exists
    if (access(workload_filename.c_str(), R_OK) < 0) {
        string error_msg;
        if (errno == ENOENT) {
            error_msg = "workload file: " + workload_filename + " does not exist";
        } else if (errno == EACCES) {
            error_msg = "workload file: " + workload_filename + " exists but is not readable";
        } else {
            error_msg = "Unknown workload file: " + workload_filename + " access error";
        }
        LoggerFactory::get_logger("workload")->critical(error_msg);
        return;
    }
  
    // there exists new workload, change the ETFeeder
    if (this->et_feeder != nullptr)
      delete this->et_feeder;
    this->et_feeder = new ETFeeder(workload_filename);
    iteration++;
    is_finished = false;
    // fire next iteration
    fire();
}

void Workload::sleep_workload(const std::vector<Sys*>& systems) {

    // add new workload filename to pending workloads in managed systems
    for (auto* managed_sys : systems) {
        if (managed_sys == nullptr || managed_sys->workload == nullptr) {
            LoggerFactory::get_logger("workload")
                ->critical("Null system or workload while sleeping workload");
        }
        // make managed systems sleep
        managed_sys->workload->is_sleep = true;
    }
    // sleep myself
    is_sleep = true;
}

void Workload::fire() {
    call(EventType::General, NULL);
}

void Workload::report() {
    Tick curr_tick = Sys::boostedTick();
    LoggerFactory::get_logger("workload")
        ->info("sys[{}] iteration {} finished, {} cycles, exposed communication {} cycles.",
               sys->id, iteration, curr_tick, curr_tick - hw_resource->tics_gpu_ops);
}
