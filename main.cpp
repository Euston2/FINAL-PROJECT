
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <IPv4Layer.h>
#include <Packet.h>
#include <PcapFileDevice.h>
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
.
constexpr int kFlushEveryNPackets = 50;


struct AppState {
    std::atomic<std::uint64_t> packets{0};

    pcpp::LinkLayerType linkType = pcpp::LINKTYPE_ETHERNET;
    std::filesystem::path outputDir;
    int rotationSeconds = 60;

    pcpp::PcapFileWriterDevice* writer = nullptr;
    std::chrono::steady_clock::time_point fileOpenedAt;
    int packetsSinceFlush = 0;
};


std::filesystem::path makeRotatedFilename(const std::filesystem::path& outputDir) {
    const std::time_t now = std::time(nullptr);
    std::tm utcTm{};
#ifdef _WIN32
    gmtime_s(&utcTm, &now);
#else
    gmtime_r(&now, &utcTm);
#endif
    std::ostringstream oss;
    oss << "edems_" << std::put_time(&utcTm, "%Y%m%dT%H%M%SZ") << ".pcap";
    return outputDir / oss.str();
}

.
bool rotateFile(AppState& state) {
    if (state.writer != nullptr) {
        state.writer->close();  
        delete state.writer;
        state.writer = nullptr;
    }

    const std::filesystem::path path = makeRotatedFilename(state.outputDir);
    state.writer = new pcpp::PcapFileWriterDevice(path.string(), state.linkType);
    if (!state.writer->open()) {
        std::cerr << "Failed to open pcap file: " << path.string() << std::endl;
        delete state.writer;
        state.writer = nullptr;
        return false;
    }

    std::cout << "[rotate] writing to " << path.string() << std::endl;
    state.fileOpenedAt = std::chrono::steady_clock::now();
    state.packetsSinceFlush = 0;
    return true;
}

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
    auto* state = static_cast<AppState*>(cookie);
    state->packets.fetch_add(1, std::memory_order_relaxed);

    
    const auto elapsed = std::chrono::steady_clock::now() - state->fileOpenedAt;
    if (elapsed >= std::chrono::seconds(state->rotationSeconds)) {
        rotateFile(*state);
    }

    
    if (state->writer != nullptr) {
        state->writer->writePacket(*packet);
        state->packetsSinceFlush++;

        
        if (state->packetsSinceFlush >= kFlushEveryNPackets) {
            state->writer->flush();
            state->packetsSinceFlush = 0;
        }
    }

    
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
        std::cout << "Usage: " << argv[0] << " <interface name> [rotation seconds, default 60]\n"
                  << "       " << argv[0] << " --list\n"
                  << "Available interfaces:\n";
        listInterfaces();
        return argc < 2 ? 1 : 0;
    }

    const std::string ifaceName = argv[1];
    const int rotationSeconds = (argc >= 3) ? std::stoi(argv[2]) : 60;

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

    AppState state;
    state.linkType = dev->getLinkType();
    state.rotationSeconds = rotationSeconds;
    state.outputDir = "captures";
    std::filesystem::create_directories(state.outputDir);

    
    if (!rotateFile(state)) {
        dev->close();
        return 1;
    }

    if (!dev->startCapture(onPacketArrives, &state)) {
        std::cerr << "Failed to start capture." << std::endl;
        state.writer->close();
        delete state.writer;
        dev->close();
        return 1;
    }

    std::cout << "Capturing on " << dev->getName() << ", rotating every " << rotationSeconds
              << "s (Ctrl+C to stop)\n"
              << std::endl;

    std::uint64_t total = 0;
    int seconds = 0;
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        const std::uint64_t perSecond = state.packets.exchange(0);
        total += perSecond;
        std::cout << "---- [" << ++seconds << "s] " << perSecond << " pkts/s (total " << total
                  << ") ----" << std::endl;
    }

    dev->stopCapture();
    dev->close();

    
    if (state.writer != nullptr) {
        state.writer->close();  // flushes and closes the final, possibly-partial file
        delete state.writer;
    }

    total += state.packets.exchange(0);
    std::cout << "\nStopped. Total packets captured: " << total << std::endl;
    return 0;
}