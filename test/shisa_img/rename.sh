#!/bin/bash

# make directory for undefined file
if [ -d "./undef" ]; then
  :
else
  mkdir ./undef
fi

# for all files
num=1
for i in *
do
  # new file name
  nfn=`printf "shisa_%05d" $num`
  # for directory
  if [ -d "$i" ]; then continue; fi
  # get the file type
  typ=`file "$i" | cut -d ":" -f 2 | cut -d " " -f 2`
  # rename
  if   [ "$typ" = "JPEG" ]; then
    mv ./"$i" ./"${nfn}.jpg"
    :
  elif [ "$typ" = "PNG" ]; then
    convert ./"$i" ./"${nfn}.jpg"
    rm ./"$i"
  elif [ "$typ" = "GIF" ]; then
    convert ./"$i" ./"${nfn}.jpg"
    rm ./"$i"
  else
    mv ./"$i" ./undef/
    continue
  fi
  num=$((num+1))
done
