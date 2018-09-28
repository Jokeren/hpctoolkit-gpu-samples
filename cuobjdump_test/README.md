## Report cuobjdump incorrect binary encoding for dual-issued instructions                                                                                     

### Observations

    /*0088*/                   IADD R4.CC, R6.reuse, c[0x0][0x140];                        /* 0x4c10800005070604 */
    /*0090*/                   IADD.X R5, R0.reuse, c[0x0][0x144];                         /* 0x4c10080005170005 */
    /*0098*/         {         IADD R2.CC, R6, c[0x0][0x148];                              /* 0x4c10800005270602 */
    /*00a8*/                   LDG.E R4, [R4];        }                                    /* 0x001dc400fc400776 */
                                                                                           /* 0xeed4200000070404 */
    /*00b0*/                   IADD.X R3, R0, c[0x0][0x14c];                               /* 0x4c10080005370003 */
    /*00b8*/                   LDG.E R2, [R2];                                             /* 0xeed4200000070202 */

In the above list, instruction *0098* and instruction *00a8* are dual-issued. Though the encoding of instruction *00a8* is displayed as `0x001dc400fc400776`, its actual encoding should be `0xeed4200000070404`, as `0x001dc400fc400776` is obviously a control code. I know this problem may not be considered as a bug because the right-most column may need to be consistent with the actual binary order.

### Usage

    make
    cuobjdump --dump-sass vecAdd.cubin
