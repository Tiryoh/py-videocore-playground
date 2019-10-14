#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


@qpu
def basic_temp(asm):
    """
    kernel program
    """
    nop()  # do nothing
    exit()  # exit GPU code


if __name__ == '__main__':
    with Driver() as drv:
        """
        host program
        """
        out = drv.alloc(16, 'float32')  # allocate GPU memory
        out[:] = 0.0

        """
        execute kernel program
        """
        drv.execute(
            n_threads=1,  # thread number for QPU (1 to 12)
            program=drv.program(basic_temp),  # the kernel program to execute
            uniforms=[out.address]  # an initial value for uniform
        )
