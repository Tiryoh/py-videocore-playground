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
    # Memory -> VPM, 16 elements * 2 lines
    setup_dma_load(nrows=2)
    start_dma_load(uniform)
    wait_dma_load()
    
    # VPM -> Register, 16 elements * 2 lines
    setup_vpm_read(nrows = 2)
    mov(r0, vpm)
    mov(r1, vpm)

    # Register -> VPM, 16 elements * 2 lines
    setup_vpm_write()
    mov(vpm, r1)
    mov(vpm, r0)

    # VPM -> Memory, 16 elements * 2 lines
    setup_dma_store(nrows=2)
    start_dma_store(uniform)
    wait_dma_store()

    exit()


if __name__ == '__main__':
    with Driver() as drv:
        """
        host program
        """
        list_a = drv.alloc((2,16), 'float32')
        for i in range(0,16):
            list_a[0][i] = 16 - i
            list_a[1][i] = 2.0 * i

        out = drv.alloc((2, 16), 'float32')  # allocate GPU memory
        out[:] = 0.0

        print(' list_a '.center(80, '='))
        print(list_a)

        """
        execute kernel program
        """
        drv.execute(
            n_threads=1,
            program=drv.program(kernel),
            uniforms=[list_a.address, out.address]
        )

        print(' out '.center(80, '='))
        print(out)
