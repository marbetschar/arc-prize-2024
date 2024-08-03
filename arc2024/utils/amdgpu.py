import os
import platform


def is_amdgpu_on_linux():
    """
    Source: https://github.com/JeLuF/stable-diffusion-ui/blob/main/scripts/check_modules.py
    :return:
    """
    if platform.system() == "Linux":
        try:
            with open("/proc/bus/pci/devices", "r") as f:
                device_info = f.read()
                if "amdgpu" in device_info and "nvidia" not in device_info:
                    return True
        except:
            return False

    return False


def setup_amdgpu_environment(
        hsa_override_gfx_version: str = "10.3.0",
        hip_visible_devices: int = 0,
        force_full_precision: bool = False
):
    """
    In order for this to work properly, make sure you have ROCm installed on your system (for example
    by using a specific PyTorch version with ROCm support - see https://pytorch.org/get-started/locally/).

    Then run `rocminfo` from the command line to retrieve information about your AMD GPU. We are interested
    in the following information: `Name: gfx1032` and `Node: 1`.

    Source: https://github.com/JeLuF/stable-diffusion-ui/blob/main/scripts/check_modules.py

    :param hsa_override_gfx_version: exact gfx version from `rocminfo` output in semver format: `gfx1032` -> `10.3.2`
    :param hip_visible_devices: Use `node minus 1` from `rocminfo` output here
    :param force_full_precision: Only set this to `True` for the older `Navi 1` GPU architecture
    :return:
    """
    if not os.access("/dev/kfd", os.W_OK):
        print(
            "No write access to /dev/kfd.",
            "Without this, the ROCm driver will probably not be able to initialize the GPU.",
            "Follow the instructions on this site to configure the access:",
            "https://github.com/easydiffusion/easydiffusion/wiki/AMD-on-Linux#access-permissions"
        )

    os.environ["HSA_OVERRIDE_GFX_VERSION"] = hsa_override_gfx_version
    os.environ["FORCE_FULL_PRECISION"] = "yes" if force_full_precision else "no"
    os.environ["HIP_VISIBLE_DEVICES"] = str(hip_visible_devices)
