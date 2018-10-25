## Report nvdisasm (cuda-9.2 and cuda-10.0) loses edge information in generated dot graphs

### Observations

*sm_60* cubins compiled and dumped by *cuda-9.2* and *cuda-10.0* toolkits lose edge port labels in the instructions.

    node [fontname="Courier",fontsize=10,shape=Mrecord];
    "vecAdd"
    [label="{<entry>  .global   vecAdd\l  .type   vecAdd,@function\l  .size   vecAdd,(.L_49\ -\ vecAdd)\l .other    vecAdd,\<no\ object\>\lvecAdd:\l.text.vecAdd:
    \l0008:\ \ \ MOV\ R1,\ c\[0x0\]\[0x20\]\ ;\l0010:\ \{\ IADD\ RZ.CC,\ -RZ,\ c\[0x0\]\[0x160\]\ ;\l/*0018*/\ \ \ S2R\ R0,\ SR_CTAID.X\ \}
    \l0028:\ \ \ ISETP.NE.X.AND\ P0,\ PT,\ RZ,\ c\[0x0\]\[0x164\],\ PT\ ;\l0030:\ \ \ S2R\ R2,\ SR_TID.X\ ;\l0038:\ \ \ XMAD\ R2,\ R0.reuse,\ c\[0x0\]\ \[0x8\],\ R2\ ;
    \l0048:\ \ \ XMAD.MRG\ R3,\ R0.reuse,\ c\[0x0\]\ \[0x8\].H1,\ RZ\ ;\l0050:\ \ \ XMAD.PSL.CBCC\ R0,\ R0.H1,\ R3.H1,\ R2\ ;\l0058:\ \ \ SHL\ R2,\ R0.reuse,\ 0x2\ ;
    \l0068:\ \ \ SHR.U32\ R3,\ R0,\ 0x1e\ ;\l0070:\ \ \ IADD\ R6.CC,\ R2.reuse,\ c\[0x0\]\[0x140\]\ ;\l0078:\ \ \ IADD.X\ R7,\ R3.reuse,\ c\[0x0\]\[0x144\]\ ;
    \l0088:\ \ \ IADD\ R4.CC,\ R2.reuse,\ c\[0x0\]\[0x148\]\ ;\l0090:\ \ \ IADD.X\ R5,\ R3.reuse,\ c\[0x0\]\[0x14c\]\ ;\l0098:\ \ \ IADD\ R2.CC,\ R2,\ c\[0x0\]\[0x150\]\ ;
    \l00a8:\ \{\ IADD.X\ R3,\ R3,\ c\[0x0\]\[0x154\]\ ;\l/*00b0*/\ \ \ @!P0\ BRA\ `(.L_1)\ \}\l00b8:\ \ \ MOV\ R8,\ c\[0x0\]\[0x160\]\ ;
    \l00c8:\ \ \ MOV\ R10,\ RZ\ ;\l00d0:\ \ \ MOV\ R12,\ RZ\ ;\l00d8:\ \ \ LOP32I.AND\ R8,\ R8,\ 0x3\ ;\l00e8:\ \ \ IADD\ RZ.CC,\ -RZ,\ R8\ ;\l00f0:\ \ \ ISETP.NE.X.AND\ P0,\ PT,\ RZ,\ RZ,\ PT\ ;
    \l00f8:\ \ \ @!P0\ BRA\ `(.L_2)\ ;\l0108:\ \ \ IADD32I\ RZ.CC,\ R8,\ -0x1\ ;\l0110:\ \ \ ISETP.NE.X.AND\ P0,\ PT,\ RZ,\ RZ,\ PT\ ;\l0118:\ \ \ @!P0\ BRA\ `(.L_3)\ ;
    \l0128:\ \{\ IADD32I\ RZ.CC,\ R8,\ -0x2\ ;\l/*0130*/\ \ \ SSY\ `(.L_4)\ \}\l0138:\ \ \ MOV32I\ R10,\ 0x1\ ;\l0148:\ \ \ MOV\ R12,\ RZ\ ;\l0150:\ \ \ ISETP.NE.X.AND\ P0,\ PT,\ RZ,\ RZ,\ PT\ ;
    \l0158:\ \ \ @!P0\ SYNC\ \ (*\"TARGET=\ .L_4\ \"*);\l0168:\ \ \ IADD\ RZ.CC,\ R0,\ -c\[0x0\]\[0x158\]\ ;\l0170:\ \ \ ISETP.GE.U32.X.AND\ P0,\ PT,\ RZ,\ c\[0x0\]\[0x15c\],\ PT\ ;
    \l0178:\ \{\ MOV32I\ R10,\ 0x2\ ;\l/*0188*/\ \ \ @P0\ SYNC\ \ (*\"TARGET=\ .L_4\ \"*)\}\l0190:\ \ \ LDG.E\ R8,\ \[R6\]\ ;\l0198:\ \ \ LDG.E\ R9,\ \[R4\]\ ;
    \l01a8:\ \ \ CAL\ `($vecAdd$_Z3addii)\ ;\l01b0:\ \ \ STG.E\ \[R2\],\ R11\ ;\l01b8:\ \ \ SYNC\ \ (*\"TARGET=\ .L_4\ \"*);\l}"]                                                                                                    
    "vecAdd":exit0:e -> ".L_1":entry:n [style=solid];
    "vecAdd":exit1:e -> ".L_2":entry:n [style=solid];
    "vecAdd":exit2:e -> ".L_3":entry:n [style=solid];
    "vecAdd":exit3:e -> ".L_4":entry:n [style=solid];
    "vecAdd":exit4:e -> ".L_4":entry:n [style=solid];
    "vecAdd":exit5:e -> ".L_4":entry:n [style=solid];
    
For instance, in the above block, `exit0-5` labels are missing in the instructions.

### Usage

    ./run.sh
    check sm_60_cuda_10.0.dot and sm_60_cuda_9.2.dot
