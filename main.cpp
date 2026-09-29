

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <IPv4Layer.h>
#include <Packet.h>
#include <PcapLiveDevice.h>
#include <PcapLiveDeviceList.h>
#include <RawPacket.h>
#include <TcpLayer.h>
#include <UdpLayer.h>

namespace {

std::atomic<bool> g_running{true};

void onSigInt(int ) {
    g_running = false;
}

struct CaptureStats {
    std::atomic<std::uint64_t> packets{0};
};


std::string formatTimestamp(const pcpp::RawPacket* packet) {
    const timespec ts = packet->getPacketTimeStamp();
    const std::time_t seconds = ts.tv_sec;
    const int milliseconds = static_cast<int>(ts.tv_nsec / 1000000);

    std::tm localTm{};
#ifdef _WIN32
    localtime_s(&localTm, &seconds);
#else
    localtime_r(&seconds, &localTm);
#endif

    std::ostringstream oss;
    oss << std::put_time(&localTm, "%H:%M:%S") << '.' << std::setfill('0') << std::setw(3)
        << milliseconds;
    return oss.str();
}


void onPacketArrives(pcpp::RawPacket* packet, pcpp::PcapLiveDevice* /*dev*/, void* cookie) {
    auto* stats = static_cast<CaptureStats*>(cookie);
    stats->packets.fetch_add(1, std::memory_order_relaxed);

    
    pcpp::Packet parsedPacket(packet);

    
    auto* ipLayer = parsedPacket.getLayerOfType<pcpp::IPv4Layer>();
    if (ipLayer == nullptr) {
        std::cout << "[" << formatTimestamp(packet) << "] non-IPv4 packet (e.g. ARP/IPv6), skipping"
                  << std::endl;
        return;
    }

    const std::string srcIp = ipLayer->getSrcIPAddress().toString();
    const std::string dstIp = ipLayer->getDstIPAddress().toString();

    
    std::string protocol;
    std::uint16_t srcPort = 0;
    std::uint16_t dstPort = 0;

    if (auto* tcpLayer = parsedPacket.getLayerOfType<pcpp::TcpLayer>()) {
        protocol = "TCP";
        srcPort = tcpLayer->getSrcPort();
        dstPort = tcpLayer->getDstPort();
    } else if (auto* udpLayer = parsedPacket.getLayerOfType<pcpp::UdpLayer>()) {
        protocol = "UDP";
        srcPort = udpLayer->getSrcPort();
        dstPort = udpLayer->getDstPort();
    } else {
        protocol = "OTHER";
    }

    std::cout << "[" << formatTimestamp(packet) << "] " << protocol << "  " << srcIp << ":" << srcPort
              << " -> " << dstIp << ":" << dstPort << std::endl;
}

void listInterfaces() {
    const auto& devList = pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDevicesList();
    for (pcpp::PcapLiveDevice* dev : devList) {
        std::cout << "Name: " << dev->getName() << std::endl;
        std::cout << "  Description: " << dev->getDesc() << std::endl;

        pcpp::IPv4Address ip4 = dev->getIPv4Address();
        if (ip4 != pcpp::IPv4Address::Zero) {
            std::cout << "  IPv4: " << ip4.toString() << std::endl;
        }
        pcpp::IPv6Address ip6 = dev->getIPv6Address();
        if (ip6 != pcpp::IPv6Address::Zero) {
            std::cout << "  IPv6: " << ip6.toString() << std::endl;
        }
    }
}

}  

int main(int argc, char* argv[]) {
    if (argc < 2 || std::string(argv[1]) == "--list") {
        std::cout << "Usage: " << argv[0] << " <interface name>   (or --list)\n"
                  << "Available interfaces:\n";
        listInterfaces();
        return argc < 2 ? 1 : 0;
    }

    const std::string ifaceName = argv[1];

    pcpp::PcapLiveDevice* dev =
        pcpp::PcapLiveDeviceList::getInstance().getPcapLiveDeviceByName(ifaceName);
    if (dev == nullptr) {
        std::cerr << "No interface named '" << ifaceName << "'. Run with --list to see valid names."
                  << std::endl;
        return 1;
    }

    if (!dev->open()) {
        std::cerr << "Failed to open '" << ifaceName
                  << "'. Try running from an Administrator terminal." << std::endl;
        return 1;
    }

    std::signal(SIGINT, onSigInt);

    CaptureStats stats;
    if (!dev->startCapture(onPacketArrives, &stats)) {
        std::cerr << "Failed to start capture." << std::endl;
        dev->close();
        return 1;
    }

    std::cout << "Capturing on " << dev->getName() << " (Ctrl+C to stop)\n" << std::endl;

    std::uint64_t total = 0;
    int seconds = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const std::uint64_t perSecond = stats.packets.exchange(0);
        total += perSecond;
        std::cout << "---- [" << ++seconds << "s] " << perSecond << " pkts/s (total " << total
                  << ") ----" << std::endl;
    }

    dev->stopCapture();
    dev->close();

    total += stats.packets.exchange(0);
    std::cout << "\nStopped. Total packets captured: " << total << std::endl;
    return 0;
}