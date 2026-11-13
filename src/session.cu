#define main original_worker_main
#include "driver.cu"
#undef main

int main(int argc, char** argv)
{
    std::string fixed_prefix;
    unsigned requests = 0;

    auto handle = [&](const std::string& request_path) -> int
    {
        try
        {
            std::ifstream input(request_path);
            if (!input)
                throw std::runtime_error("cannot open session request");
            std::string line;
            std::getline(input, line);
            const int request_argc = std::stoi(line);
            if (request_argc != 11)
                throw std::runtime_error("session expects GraphCut arguments");
            std::vector<std::string> args(request_argc);
            for (auto& arg : args)
                if (!std::getline(input, arg))
                    throw std::runtime_error("truncated session arguments");
            if (fixed_prefix.empty())
                fixed_prefix = args[1];
            if (fixed_prefix != args[1])
                throw std::runtime_error("cannot change shard with live singleton cache");
            std::getline(input, line);
            const int count = std::stoi(line);
            if (count < 0 || count > 256)
                throw std::runtime_error("invalid session environment count");
            for (const char* key : {"BPG_EMBEDDED_ROUTER_CENTROIDS", "BPG_EMBEDDED_ROUTER_OWNERS",
                                    "BPG_EMBEDDED_ROUTER_MODE", "BPG_EMBEDDED_ROUTER_RATIO",
                                    "BPG_EMBEDDED_ROUTER_START", "BPG_EMBEDDED_ROUTER_DONE",
                                    "BPG_EMBEDDED_ROUTER_OUTPUT", "BPG_BATCH_CPU_LISTS"})
                ::unsetenv(key);
            for (int i = 0; i < count; ++i)
            {
                if (!std::getline(input, line))
                    throw std::runtime_error("truncated session environment");
                const auto pos = line.find('=');
                if (pos == std::string::npos || pos == 0)
                    throw std::runtime_error("invalid environment line");
                const auto key = line.substr(0, pos);
                const auto value = line.substr(pos + 1);
                if (key.find("BPG_") != 0 && key.find("PHMS_") != 0 && key.find("OMP_") != 0)
                    throw std::runtime_error("unexpected session environment key");
                ::setenv(key.c_str(), value.c_str(), 1);
            }
            std::vector<char*> request_argv;
            for (auto& arg : args)
                request_argv.push_back(&arg[0]);
            request_argv.push_back(nullptr);
            const int result = original_worker_main(request_argc, request_argv.data());
            if (result)
                return result;
            ++requests;
            std::cout << "SESSION_REQUEST_DONE " << requests << " " << request_path << std::endl;
            std::ofstream done(request_path + ".done");
            done << "requests=" << requests << "\n";
        }
        catch (const std::exception& e)
        {
            std::cerr << "session_worker: " << e.what() << std::endl;
            return 1;
        }
        return 0;
    };

    if (argc > 1)
    {
        for (int i = 1; i < argc; ++i)
        {
            const int result = handle(argv[i]);
            if (result)
                return result;
        }
        return 0;
    }

    std::string request_path;
    while (std::getline(std::cin, request_path))
    {
        if (request_path == "QUIT")
            return 0;
        const int result = handle(request_path);
        if (result)
            return result;
    }
    return 0;
}
