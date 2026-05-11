"""
fs_x8x.py

gem5 X86 full-system config for an X86 host and X86 guest.
The VM boots with KVM cores and then switches to Timing cores after boot.

Important:
- This script uses locally defined filesystem paths for the kernel and disk image.
- The disk image must be gem5-ready for post-boot scripting, i.e. it must:
  1) contain the `m5` utility in the guest, and
  2) execute `m5 readfile` (or equivalent gem5 after-boot hook) after login/boot.
  Without that, the `readfile_contents` script below will never run and the
  automatic KVM -> Timing switch will not be triggered.
- X86Board currently expects memory <= 3GiB.
"""

from pathlib import Path

from gem5.components.boards.x86_board import X86Board
from gem5.components.cachehierarchies.classic.private_l1_private_l2_walk_cache_hierarchy import (
    PrivateL1PrivateL2WalkCacheHierarchy,
)
from gem5.components.memory.single_channel import SingleChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.isas import ISA
from gem5.resources.resource import DiskImageResource, KernelResource
from gem5.simulate.simulator import Simulator
from gem5.utils.requires import requires

from gem5.simulate.exit_handler import ExitHandler
from gem5.simulate.simulator import Simulator
from gem5.utils.override import overrides


# -----------------------------------------------------------------------------
# Local resource paths: edit these for your setup.
# -----------------------------------------------------------------------------
KERNEL_PATH = "./vmlinux-x86-ubuntu-6.8.0-52-generic"
DISK_IMAGE_PATH = "./ubuntu-2404-loaded"

# Most gem5 Ubuntu images boot from partition 1. Change this if your image uses
# a different root partition.
ROOT_PARTITION = "1"

# Keep X86Board memory <= 3GiB.
MEMORY_SIZE = "3GiB"
CLK_FREQ = "3GHz"
NUM_CORES = 1


# -----------------------------------------------------------------------------
# Basic host / binary checks.
# -----------------------------------------------------------------------------
requires(
    isa_required=ISA.X86,
    kvm_required=True,
)

kernel_path = Path(KERNEL_PATH)
disk_image_path = Path(DISK_IMAGE_PATH)

if not kernel_path.is_file():
    raise FileNotFoundError(f"Kernel not found: {kernel_path}")

if not disk_image_path.is_file():
    raise FileNotFoundError(f"Disk image not found: {disk_image_path}")


# -----------------------------------------------------------------------------
# System components.
# -----------------------------------------------------------------------------
cache_hierarchy = PrivateL1PrivateL2WalkCacheHierarchy(
    l1d_size="32KiB",
    l1i_size="32KiB",
    l2_size="256KiB",
)

memory = SingleChannelDDR4_2400(size=MEMORY_SIZE)

processor = SimpleSwitchableProcessor(
    starting_core_type=CPUTypes.KVM,
    switch_core_type=CPUTypes.TIMING,
    isa=ISA.X86,
    num_cores=NUM_CORES,
)

# Avoid perf-based KVM support requirements on the host.
for proc in processor.start:
    proc.core.usePerf = False

board = X86Board(
    clk_freq=CLK_FREQ,
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)


# -----------------------------------------------------------------------------
# Local resources.
# -----------------------------------------------------------------------------
kernel = KernelResource(
    local_path=str(kernel_path),
    architecture=ISA.X86,
)

disk_image = DiskImageResource(
    local_path=str(disk_image_path),
    root_partition=ROOT_PARTITION,
)


AFTER_BOOT_SCRIPT = r"""
#!/bin/bash
echo '12345' | sudo -S chmod +rx ./gpt2small_x86_v6
printf "\n===Binary start===\n"
./gpt2small_x86_v6
printf "===Binary end===\n"
"""

board.set_kernel_disk_workload(
    kernel=kernel,
    disk_image=disk_image,
    readfile_contents=AFTER_BOOT_SCRIPT,
        kernel_args=[
            "earlyprintk=ttyS0",
            "console=ttyS0",
            "root=/dev/sda2",
        ]
            #add "interactive" for interactive
)


# -----------------------------------------------------------------------------
# Exit handling.
# -----------------------------------------------------------------------------
class KernelBootedExitHandler(ExitHandler, hypercall_num=1):
    @overrides(ExitHandler)
    def _process(self, simulator: "Simulator") -> None:
        print("Kernel has booted")

    @overrides(ExitHandler)
    def _exit_simulation(self) -> bool:
        return False


class AfterBootExitHandler(ExitHandler, hypercall_num=2):
    @overrides(ExitHandler)
    def _process(self, simulator: "Simulator") -> None:
        print("System is in after_boot.sh")
        simulator.switch_processor()

    @overrides(ExitHandler)
    def _exit_simulation(self) -> bool:
        return False


class ScriptDoneExitHandler(ExitHandler, hypercall_num=3):
    @overrides(ExitHandler)
    def _process(self, simulator: "Simulator") -> None:
        print("The script is done")
        print("Quitting simulation")

    @overrides(ExitHandler)
    def _exit_simulation(self) -> bool:
        return True


simulator = Simulator(
    board=board,
)

simulator.run()
