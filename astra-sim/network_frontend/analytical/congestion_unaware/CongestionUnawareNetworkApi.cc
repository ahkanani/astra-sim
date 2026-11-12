/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_unaware/CongestionUnawareNetworkApi.hh"
#include <cassert>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <astra-network-analytical/congestion_unaware/Switch.h>
#include <astra-network-analytical/congestion_unaware/FullyConnected.h>
#include <astra-network-analytical/congestion_unaware/Ring.h>
#include <astra-network-analytical/common/NetworkFunction.h>

using namespace AstraSim;
using namespace AstraSimAnalyticalCongestionUnaware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionUnaware;

std::shared_ptr<Topology> CongestionUnawareNetworkApi::topology;
std::shared_ptr<Topology> CongestionUnawareNetworkApi::replay_scaleout;
std::vector<int> CongestionUnawareNetworkApi::replay_domains;
std::vector<double> CongestionUnawareNetworkApi::replay_tx_free;
std::vector<double> CongestionUnawareNetworkApi::replay_rx_free;
double CongestionUnawareNetworkApi::replay_bandwidth = 0;
double CongestionUnawareNetworkApi::replay_local_bandwidth = 0;
std::vector<double> CongestionUnawareNetworkApi::replay_local_tx_free;
std::vector<double> CongestionUnawareNetworkApi::replay_local_rx_free;

void CongestionUnawareNetworkApi::set_replay_domains(
    const std::vector<int>& domains, const std::string& kind, double bandwidth, double latency,
    double local_bandwidth) {
    if (domains.empty() || bandwidth <= 0 || latency < 0 ||
        *std::min_element(domains.begin(), domains.end()) < 0)
        throw std::runtime_error("Invalid replay network domains");
    const auto count = *std::max_element(domains.begin(), domains.end()) + 1;
    replay_domains = domains;
    // Match the analytical library's existing GiB/s conversion, including
    // when the user supplies a per-dimension configuration vector.
    replay_bandwidth = bw_GBps_to_Bpns(bandwidth);
    replay_tx_free.assign(count, 0);
    replay_rx_free.assign(count, 0);
    replay_local_bandwidth = local_bandwidth > 0 ? bw_GBps_to_Bpns(local_bandwidth) : 0;
    replay_local_tx_free.assign(domains.size(), 0);
    replay_local_rx_free.assign(domains.size(), 0);
    if (kind == "Switch") replay_scaleout = std::make_shared<Switch>(count, bandwidth, latency);
    else if (kind == "FullyConnected") replay_scaleout = std::make_shared<FullyConnected>(count, bandwidth, latency);
    else if (kind == "Ring") replay_scaleout = std::make_shared<Ring>(count, bandwidth, latency);
    else throw std::runtime_error("Invalid replay scale-out topology");
}

void CongestionUnawareNetworkApi::set_topology(
    std::shared_ptr<Topology> topology_ptr) noexcept {
    assert(topology_ptr != nullptr);

    // move topology
    CongestionUnawareNetworkApi::topology = std::move(topology_ptr);

    // set topology-related values
    CongestionUnawareNetworkApi::dims_count =
        CongestionUnawareNetworkApi::topology->get_dims_count();
    CongestionUnawareNetworkApi::bandwidth_per_dim =
        CongestionUnawareNetworkApi::topology->get_bandwidth_per_dim();
}

CongestionUnawareNetworkApi::CongestionUnawareNetworkApi(
    const int rank) noexcept
    : CommonNetworkApi(rank) {
    assert(rank >= 0);
}

int CongestionUnawareNetworkApi::sim_send(void* const buffer,
                                          const uint64_t count,
                                          const int type,
                                          const int dst,
                                          const int tag,
                                          sim_request* const request,
                                          void (*msg_handler)(void*),
                                          void* const fun_arg) {
    // query chunk id
    const auto src = sim_comm_get_rank();
    const auto chunk_id =
        CongestionUnawareNetworkApi::chunk_id_generator.create_send_chunk_id(
            tag, src, dst, count);

    // search tracker
    const auto entry =
        callback_tracker.search_entry(tag, src, dst, count, chunk_id);
    if (entry.has_value()) {
        // recv operation already issued.
        // add send event handler to the tracker
        entry.value()->register_send_callback(msg_handler, fun_arg);
    } else {
        // recv operation not issued yet
        // create new entry and insert send callback
        auto* const new_entry =
            callback_tracker.create_new_entry(tag, src, dst, count, chunk_id);
        new_entry->register_send_callback(msg_handler, fun_arg);
    }

    // create chunk
    auto chunk_arrival_arg = std::tuple(tag, src, dst, count, chunk_id);
    auto arg = std::make_unique<decltype(chunk_arrival_arg)>(chunk_arrival_arg);
    const auto arg_ptr = static_cast<void*>(arg.release());

    // compute send communication delay (in AstraSim format)
    auto send_delay_ns = topology->send(src, dst, count);
    if (!replay_domains.empty()) {
        const auto source = replay_domains.at(src);
        const auto target = replay_domains.at(dst);
        if (source != target) {
            // Reuse the configured outer topology. All ranks in a GPU domain
            // share its full-duplex NIC with GPU collectives and PIM traffic.
            const auto delay = replay_scaleout->send(source, target, count);
            const auto serialization = static_cast<double>(count) / replay_bandwidth;
            const auto propagation = std::max(0.0, static_cast<double>(delay) - serialization);
            const auto now = static_cast<double>(event_queue->get_current_time());
            const auto start = std::max({now, replay_tx_free[source], replay_rx_free[target] - propagation});
            replay_tx_free[source] = start + serialization;
            replay_rx_free[target] = start + propagation + serialization;
            send_delay_ns = static_cast<EventTime>(std::ceil(start - now + delay));
        } else if (replay_local_bandwidth > 0) {
            // A placement-aware local exchange can fan out from a domain
            // leader. Do not grant a full GPU link to each peer independently.
            const auto serialization = static_cast<double>(count) / replay_local_bandwidth;
            const auto propagation = std::max(0.0, static_cast<double>(send_delay_ns) - serialization);
            const auto now = static_cast<double>(event_queue->get_current_time());
            const auto start = std::max({now, replay_local_tx_free[src],
                                        replay_local_rx_free[dst] - propagation});
            replay_local_tx_free[src] = start + serialization;
            replay_local_rx_free[dst] = start + propagation + serialization;
            send_delay_ns = static_cast<EventTime>(std::ceil(start - now + send_delay_ns));
        }
    }
    const auto send_delay = static_cast<double>(send_delay_ns);
    const auto delta = timespec_t({NS, send_delay});

    // register chunk arrival event after send communication delay
    sim_schedule(delta, CongestionUnawareNetworkApi::process_chunk_arrival,
                 arg_ptr);

    // return
    return 0;
}
