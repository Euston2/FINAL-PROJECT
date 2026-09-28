#include <iostream>
#include <PcapLiveDeviceList.h>
#include <PcapLiveDevice.h>

int main() {
    std::cout << "--- EDEMS Network Interfaces ---" << std::endl;

    // Retrieve the singleton instance of the device list and get the devices
    const std::vector<pcpp::PcapLiveDevice*>& devList = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();

    if (devList.empty()) {
        std::cerr << "No network interfaces found. Make sure Npcap is installed and running." << std::endl;
        return 1;
    }

    // Iterate over the devices and print their details
    for (pcpp::PcapLiveDevice* dev : devList) {
        // Name is usually the UUID of the interface on Windows (e.g. \Device\NPF_{...})
        std::cout << "Name: " << dev->getName() << std::endl;
        // Description is usually a human-readable name like "Intel(R) Ethernet Connection..."
        std::cout << "Description: " << dev->getDesc() << std::endl;
        
        // getAddresses() returns raw libpcap pcap_addr_t structs (a plain sockaddr*,
        // no toString()), so we use PcapPlusPlus's own wrapped accessors instead,
        // which return pcpp::IPv4Address / pcpp::IPv6Address objects that do.
        pcpp::IPv4Address ip4 = dev->getIPv4Address();
        if (ip4 != pcpp::IPv4Address::Zero) {
            std::cout << "IPv4 Address: " << ip4.toString() << std::endl;
        }

        pcpp::IPv6Address ip6 = dev->getIPv6Address();
        if (ip6 != pcpp::IPv6Address::Zero) {
            std::cout << "IPv6 Address: " << ip6.toString() << std::endl;
        }
        std::cout << "--------------------------------" << std::endl;
    }

    return 0;
}