import os
from pathlib import Path
import pytest

# Expected path on typical systems, or pass via environment variable
SLASH_ROOT = os.environ.get("SLASH_ROOT", None)

@pytest.mark.skipif(not SLASH_ROOT or not Path(SLASH_ROOT).exists(),
                    reason="SLASH_ROOT not provided or does not exist")
def test_slash_real_directory_structure():
    """Verify that our assumptions about the SLASH repository match reality."""
    slash_root = Path(SLASH_ROOT).resolve()
    
    # 1. Abstract Shell DCP location
    dcp_path = slash_root / "submodules/v80-vitis-flow/resources/abstract_shell/abs_shell_slash.dcp"
    assert dcp_path.exists(), f"Abstract shell DCP not found at {dcp_path}"
    
    # 2. CMake Module path
    cmake_dir = slash_root / "cmake"
    assert cmake_dir.exists(), f"CMake directory not found at {cmake_dir}"
    
    # 3. Check SlashTools.cmake exists
    slash_tools = cmake_dir / "SlashTools.cmake"
    assert slash_tools.exists(), f"SlashTools.cmake not found at {slash_tools}"
    
    # 4. Check for build_hls and add_vbin definitions
    cmake_content = slash_tools.read_text(encoding="utf-8")
    assert "function(build_hls" in cmake_content or "macro(build_hls" in cmake_content, "build_hls not defined in SlashTools.cmake"
    assert "function(add_vbin" in cmake_content or "macro(add_vbin" in cmake_content, "add_vbin not defined in SlashTools.cmake"
    
    # 5. Check if the VRT runtime module exists
    vrt_dir = slash_root / "vrt"
    assert vrt_dir.exists(), f"VRT directory not found at {vrt_dir}"
    assert (vrt_dir / "CMakeLists.txt").exists(), "VRT CMakeLists.txt not found"
