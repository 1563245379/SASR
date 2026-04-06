"""
Runtime monkey-patches for third-party packages that are incompatible with NumPy 2.0.

NumPy 2.0 enforces strict overflow checks on integer types. Several properties in
`nes_py` and `gym_super_mario_bros` perform arithmetic on numpy.uint8 values that
overflow under these stricter rules. This module patches those properties at import
time so that no manual edits to installed packages are needed.

Usage:
    import SASR.compat_patches  # applies patches on import
"""

import importlib


def _patch_nes_py_rom():
    """Patch nes_py._rom.ROM to avoid uint8 overflow in size calculations."""
    try:
        mod = importlib.import_module("nes_py._rom")
    except ImportError:
        return

    ROM = mod.ROM

    @property
    def prg_rom_size(self):
        return 16 * int(self.header[4])

    @property
    def chr_rom_size(self):
        return 8 * int(self.header[5])

    ROM.prg_rom_size = prg_rom_size
    ROM.chr_rom_size = chr_rom_size


def _patch_smb_env():
    """Patch gym_super_mario_bros.smb_env.SuperMarioBrosEnv to avoid uint8 overflow."""
    try:
        mod = importlib.import_module("gym_super_mario_bros.smb_env")
    except ImportError:
        return

    cls = mod.SuperMarioBrosEnv

    @property
    def _level(self):
        return int(self.ram[0x075f]) * 4 + int(self.ram[0x075c])

    @property
    def _x_position(self):
        return int(self.ram[0x6d]) * 0x100 + int(self.ram[0x86])

    cls._level = _level
    cls._x_position = _x_position


# Apply all patches on import
_patch_nes_py_rom()
_patch_smb_env()
