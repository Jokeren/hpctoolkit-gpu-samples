## Report nvdisasm bugs for cubins generated with *-arch sm_70* flag

### Observations

Cubins with `BSSY`, `BSYNC`, and `BREAK` instructions are unable to parse by nvdisasm.

### Usage

    make
    ./run.sh
    
