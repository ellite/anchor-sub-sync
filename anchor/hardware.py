import torch
import platform
import math
import os
import sys
import subprocess
from rich.console import Console

# Try to import psutil for accurate CPU RAM detection
try:
    import psutil
except ImportError:
    psutil = None

console = Console()

def get_cpu_name():
    """Return a best-effort CPU name string."""
    # 1. Try macOS (Darwin) specific command first
    if platform.system() == "Darwin":
        try:
            command = ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
            # sysctl returns bytes, so we strip and decode
            return subprocess.check_output(command).strip().decode()
        except Exception:
            pass

    # 2. Generic platform check
    try:
        name = platform.processor()
        if name and name.strip():
            return name.strip()
    except Exception:
        pass
    
    # 3. Linux fallback
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # 4. python-cpuinfo fallback
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get("brand_raw") or info.get("brand")
        if brand:
            return brand
    except Exception:
        pass

    return "CPU"

def get_system_ram_gb():
    """Returns system RAM in GB."""
    if psutil:
        return psutil.virtual_memory().available / (1024 ** 3)
    
    if platform.system() == "Darwin":
        try:
            cmd = ["sysctl", "-n", "hw.memsize"]
            total_bytes = int(subprocess.check_output(cmd).strip())
            return total_bytes / (1024 ** 3)
        except Exception:
            pass

    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable" in line:
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            pass
            
    return 8.0

def select_model_size(memory_gb, is_gpu=True):
    """
    Selects the best Whisper model based on available memory (GB).
    """
    buffer = 1.0 if is_gpu else 2.0 
    usable_mem = memory_gb - buffer

    if usable_mem >= 10:
        return "large-v3" if is_gpu else "medium"
    elif usable_mem >= 5:
        return "medium"
    elif usable_mem >= 2:
        return "small"
    else:
        return "base"

def select_translation_model(memory_gb, is_gpu=True):
    """
    Selects the best NLLB translation model based on memory.
    Returns the HuggingFace ID for the CTranslate2 INT8 model.
    """
    # Specific Repos for CTranslate2 INT8 models
    NLLB_600M = "softcatala/nllb-200-distilled-600M-ct2-int8" # ~0.6 GB VRAM
    NLLB_1_3B = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"    # ~1.5 GB VRAM
    NLLB_3_3B = "OpenNMT/nllb-200-3.3B-ct2-int8"              # ~3.5 GB VRAM

    if not is_gpu:
        # CPU is slower, stick to the lightweight distilled model unless lots of RAM
        if memory_gb > 16:
            return NLLB_1_3B
        return NLLB_600M

    # GPU Logic
    # We apply a stricter buffer because we might want to load this alongside other things
    # or ensure smoothness.
    
    if memory_gb >= 8:
        return NLLB_3_3B # Best quality
    elif memory_gb >= 4:
        return NLLB_1_3B # Great balance (RTX 2000e target)
    else:
        return NLLB_600M # Fast

def get_compute_device(force_model=None, force_batch=None, force_translation_model=None):
    """
    Detects hardware and selects optimal settings + Whisper model + Translation model.
    """
    device = "cpu"
    compute_type = "int8"
    batch_size = 4
    model_size = "base"
    translation_model = "softcatala/nllb-200-distilled-600M-ct2-int8" # Default safe fallback
    
    # 1. Check for NVIDIA CUDA
    if torch.cuda.is_available():
        if torch.version.hip:
            console.print("[bold red]🛑 Hardware Detected:[/bold red] AMD GPU (ROCm)")
            device = "cuda"
            compute_type = "float16" 
            model_size = "medium"
            translation_model = select_translation_model(8, is_gpu=True) # Assume mid-range for AMD
        else:
            device_count = torch.cuda.device_count()
            min_mem_gb = 0
            
            try:
                mems = []
                for i in range(device_count):
                    name = torch.cuda.get_device_name(i)
                    props = torch.cuda.get_device_properties(i)
                    gb = props.total_memory / (1024 ** 3)
                    mems.append(gb)
                    console.print(f"[bold green]🚀 Hardware Detected:[/bold green] {name} ({math.ceil(gb)} GB)")
                
                if mems:
                    min_mem_gb = min(mems)
            except Exception:
                console.print("[bold green]🚀 Hardware Detected:[/bold green] NVIDIA GPU (Unknown VRAM)")
                min_mem_gb = 4 

            device = "cuda"
            compute_type = "float16"
            
            model_size = select_model_size(min_mem_gb, is_gpu=True)
            translation_model = select_translation_model(min_mem_gb, is_gpu=True)
            
            # Select Batch Size
            if min_mem_gb >= 24:
                batch_size = 32
            elif min_mem_gb >= 12:
                batch_size = 16
            elif min_mem_gb >= 8:
                batch_size = 8
            else:
                batch_size = 4

    # 2. Check for Apple Silicon (Mac)
    elif torch.backends.mps.is_available():
        # NOTE: CTranslate2 (backend of faster-whisper/NLLB) does NOT support 'mps' device yet.
        # We must fall back to CPU.
        device = "cpu"
        compute_type = "int8"
        batch_size = 8
        sys_ram = get_system_ram_gb()
        cpu_name = get_cpu_name()
        console.print(f"[bold cyan]🍎 Hardware Detected:[/bold cyan] Apple Silicon (Running on CPU {cpu_name})")
        console.print(f"[dim]   System RAM Available: {sys_ram:.1f} GB[/dim]")
        
        model_size = select_model_size(sys_ram, is_gpu=False)
        translation_model = select_translation_model(sys_ram, is_gpu=False)

    # 3. Check for Intel Arc / iGPU
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
         device = "xpu"
         compute_type = "float16"
         batch_size = 8
         console.print("[bold blue]🔵 Hardware Detected:[/bold blue] Intel Arc/XPU")
         model_size = "medium"
         translation_model = select_translation_model(8, is_gpu=True) # Assume mid-range

    # 4. CPU Fallback
    else:
        cpu_name = get_cpu_name()
        ram_gb = get_system_ram_gb()
        
        console.print(f"[bold blue]🖥️ Hardware Detected:[/bold blue] CPU ({cpu_name})")
        console.print(f"[dim]   System RAM Available: {ram_gb:.1f} GB[/dim]")

        model_size = select_model_size(ram_gb, is_gpu=False)
        translation_model = select_translation_model(ram_gb, is_gpu=False)

    # 5. User Overrides
    # Model override
    if force_model:
        valid_models = [
            "tiny", "base", "small", "medium", "large",
            "large-v1", "large-v2", "large-v3",
            "distil-large-v2", "distil-medium.en", "distil-small.en",
            "distil-large-v3", "distil-large-v3.5",
            "large-v3-turbo", "turbo",
        ]

        clean_model = force_model.lower().strip()

        if clean_model.endswith(".en") and "distil" not in clean_model:
            clean_model = clean_model[:-3]

        if clean_model not in valid_models:
            console.print(f"\n[bold red]❌ Invalid model name '{force_model}' provided![/bold red]")
            console.print(f"[dim]Valid options: {', '.join(valid_models)}[/dim]")
            sys.exit(1)
        
        if clean_model == "large":
            clean_model = "large-v3"
            
        model_size = clean_model

    # Batch Size Override
    if force_batch:
        if force_batch <= 0:
            console.print(f"\n[bold red]❌ Invalid batch size '{force_batch}' provided! Must be a positive integer.[/bold red]")
            sys.exit(1)
        batch_size = force_batch

    # Translation Model Override
    if force_translation_model:
        # 1. Define Aliases for ease of use
        aliases = {
            "small":  "softcatala/nllb-200-distilled-600M-ct2-int8",
            "medium": "softcatala/nllb-200-1.3B-ct2-int8",
            "large":  "OpenNMT/nllb-200-3.3B-ct2-int8",
        }

        # 2. Define the exact valid list (The Aliases + The Raw IDs)
        valid_map = aliases.copy()
        
        # ...and add the raw strings from our valid list so they map to themselves
        raw_valid_models = [
            "softcatala/nllb-200-distilled-600M-ct2-int8",
            "OpenNMT/nllb-200-distilled-600M-ct2-int8",
            "softcatala/nllb-200-1.3B-ct2-int8",
            "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
            "OpenNMT/nllb-200-1.3B-ct2-int8",
            "OpenNMT/nllb-200-3.3B-ct2-int8",
        ]
        
        for m in raw_valid_models:
            valid_map[m] = m

        # 3. Validation Logic
        clean_input = force_translation_model.strip()
        
        if clean_input not in valid_map:
            console.print(f"\n[bold red]❌ Invalid translation model '{clean_input}' provided![/bold red]")
            console.print(f"[dim]You can use aliases: [bold white]small, medium, large[/bold white][/dim]")
            console.print(f"[dim]Or specific CTranslate2 repo IDs like 'OpenNMT/nllb-200-3.3B-ct2-int8'[/dim]")
            sys.exit(1)
            
        translation_model = valid_map[clean_input]

    return device, compute_type, batch_size, model_size, translation_model