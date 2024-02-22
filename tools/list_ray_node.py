import argparse
import re
from typing import Dict

import ray
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description="List Nodes Available Resources for Target Ray Cluster")


@ray.remote
def _available_resources_per_node() -> Dict:
    """
    Get the current nodes' available resources.

    Returns:
        (Dict) A Dict format string representing the available resource info
            of nodes in cluster.

    """

    return ray._private.state.state._available_resources_per_node()


if __name__ == "__main__":
    # parse arguments
    parser.add_argument("--address", "-a", type=str, help="server address for ray cluster, e.g. 10.140.0.168:10001")
    parser.add_argument("--nodes", type=str, nargs="+", help="list nodes idle resources with target ip addresses")
    args = parser.parse_args()

    if "-" in args.address:
        args.address = '.'.join(args.address.split('-')[-4:])  # SH-IDC1-10-140-0-160 -> 10.140.0.160
    if not args.address.endswith(":10001"):
        args.address = f"{args.address}:10001"  # 10.140.0.160 -> 10.140.0.160:10001
    context = ray.init(address=f"ray://{args.address}")

    nodes = ray.get(_available_resources_per_node.remote())
    assert nodes

    nodes_idle_res = []
    for node_id, node_res in nodes.items():
        cur_node = {}
        node_ip = "127.0.0.1"
        node_type = "COMPUTE"
        idle_cpu = 0.0
        idle_gpu = 0.0
        gpu_type = "V100"
        idle_mem = 0.0
        idle_obj_mem = 0.0
        vc_label = ""
        for res_key, res_value in node_res.items():
            if re.search("^node:", res_key):
                node_ip = res_key.split(":")[1]
            if res_key == "HEAD":
                node_type = "HEAD"
            if res_key == "CPU":
                idle_cpu = res_value
            if res_key == "GPU":
                idle_gpu = res_value
            if re.search("^accelerator_type:", res_key):
                gpu_type = res_key.split(":")[1]
            if res_key == "memory":
                idle_mem = res_value
            if res_key == "object_store_memory":
                idle_obj_mem = res_value
            if re.search("^virtual_cluster", res_key) and "group" not in res_key:
                vc_label = res_key

        cur_node["IP"] = node_ip
        cur_node["Type"] = node_type
        cur_node["CPU"] = idle_cpu
        cur_node["GPU"] = idle_gpu
        cur_node["GPUType"] = gpu_type
        cur_node["Memory"] = idle_mem
        cur_node["ObjectStoreMemory"] = idle_obj_mem
        cur_node["VirtualCluster"] = vc_label

        nodes_idle_res.append(cur_node)

    # print(nodes_idle_res)

    # print nodes in table
    table = PrettyTable(field_names=["index", "node_ip", "node_type", "virtual_cluster", "idle_cpu", "idle_gpu"])
    for index, node in enumerate(
        sorted(nodes_idle_res, key=lambda node: (node["VirtualCluster"], node["GPU"], node["IP"]))
    ):
        if args.nodes is not None and node["IP"] not in args.nodes:
            continue
        table.add_row([index, node["IP"], node["Type"], node["VirtualCluster"], node["CPU"], node["GPU"]])

    print(table)
