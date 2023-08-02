from collections import OrderedDict
import psutil


def cpu_info():
    CPUinfo = OrderedDict()
    procinfo = OrderedDict()
    nprocs = 0
    with open("/proc/cpuinfo") as f:
        for line in f:
            if not line.strip():
                # end of one processor
                CPUinfo["proc%s" % nprocs] = procinfo
                nprocs = nprocs + 1
                # Reset
                procinfo = OrderedDict()
            else:
                if len(line.split(":")) == 2:
                    procinfo[line.split(":")[0].strip()] = line.split(":")[1].strip()
                else:
                    procinfo[line.split(":")[0].strip()] = ""
    return CPUinfo


def generate_info():
    info = ""
    for processor in cpu_info().keys():
        cpu_model = cpu_info()[processor]["model name"]
    info += "CPU: %s\n" % (cpu_model)

    try:
        import pynvml

        has_pynvml = True
    except ModuleNotFoundError:
        print("Install pynvml to get GPU info")
        has_pynvml = False
    if has_pynvml:
        info += "GPU:"
        pynvml.nvmlInit()
        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu = pynvml.nvmlDeviceGetName(handle)
            info += " " + gpu.decode("utf-8")
        info += "\n"
    else:
        info += "GPU: unable to detect; please install pynvml\n"
    mem_g = psutil.virtual_memory().total / 1024 / 1024 / 1024
    info += "memory: %d GB\n" % mem_g
    return info


if __name__ == "__main__":
    print(generate_info())
