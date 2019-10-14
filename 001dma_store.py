#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from videocore.assembler import qpu
from videocore.driver import Driver


@qpu
def dma_store(asm):
    """
    kernel program
    """
    setup_vpm_write()
    mov(vpm, 1.0)

    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()


if __name__ == '__main__':
    with Driver() as drv:
        """
        host program
        """
        out = drv.alloc(16, 'float32')  # allocate GPU memory
        out[:] = 0.0

        print(' out_Before '.center(80, '='))
        print(out)

        """
        execute kernel program
        """
        drv.execute(
            n_threads=1,
            program=drv.program(dma_store),
            uniforms=[out.address]
        )

        print(' out_After '.center(80, '='))
        print(out)
