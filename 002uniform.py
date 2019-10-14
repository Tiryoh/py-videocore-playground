#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


@qpu
def kernel(asm):
    """
    kernel program
    """
    setup_vpm_write()
    mov(vpm, uniform)
    mov(vpm, uniform)
    mov(vpm, uniform)

    setup_dma_store(nrows=3)  # VPM -> Memory, 16 elements * 3 lines
    start_dma_store(uniform)
    wait_dma_store()

    exit()


if __name__ == '__main__':
    with Driver() as drv:
        """
        host program
        """
        out = drv.alloc((3, 16), 'uint32')  # allocate GPU memory
        out[:] = 0.0

        print(' out '.center(80, '='))
        print(out)

        """
        execute kernel program
        """
        drv.execute(
            n_threads=1,
            program=drv.program(kernel),
            uniforms=[1, 2, 3, out.address]
        )

        print(' out '.center(80, '='))
        print(out)
