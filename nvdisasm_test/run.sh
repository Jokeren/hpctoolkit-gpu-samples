#!/bin/bash

for f in *.cubin
do
  nvdisasm $f > /dev/null
  retv=$?
  if [ $retv -ne 0 ]
    then
      echo "nvdisasm failed for "$f
  fi
done 
