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

    # calc: add r0 and r1
    # Register -> VPM, 16 elements * 2 lines
    setup_vpm_write()
    fadd(vpm, r0, r1)

    # VPM -> Memory, 16 elements * 1 line
    setup_dma_store(nrows=1)
    start_dma_store(uniform)
    wait_dma_store()

    exit()


if __name__ == '__main__':
    with Driver() as drv:
        """
        host program
        """
        list_a = np.arange(16).astype('float32')
        list_b = np.full(16, 2.0).astype('float32')

        # combine arrays
        inp = drv.copy(np.r_[list_a, list_b])
        out = drv.alloc(16, 'float32')

        print(' list_a '.center(80, '='))
        print(list_a)
        print(' list_b '.center(80, '='))
        print(list_b)

        """
        execute kernel program
        """
        drv.execute(
            n_threads=1,
            program=drv.program(kernel),
            uniforms=[inp.address, out.address]
        )

        print(' list_a + list_b gpu_out '.center(80, '='))
        print(out)

        cpu_ans = list_a + list_b
        error = cpu_ans - out

        print(' list_a + list_b cpu_out '.center(80, '='))
        print(cpu_ans)

        print(' cpu/gpu error '.center(80, '='))
        print(np.abs(error))
