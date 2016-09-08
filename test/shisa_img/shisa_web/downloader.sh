#!/bin/bash

# http://blog.tottokug.com/entry/2015/07/30/214048
# require jq command

# search word
SEARCHWORD="沖縄 シーサー"

# get the API key from an external file
source ./apikey

QUOTEDKEY=$(echo $APIKEY | nkf -wMQ| tr = %)
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36"
QUERY=$(echo "'${SEARCHWORD}'"| nkf -wMQ | tr = %)
URL="https://${QUOTEDKEY}:${QUOTEDKEY}@api.datamarket.azure.com/Bing/Search/v1/Image?Query=$QUERY&Market=%27ja-JP%27&\$format=json"

export LC_ALL=C

for i in `seq 1 15 30`
do
  curl --basic $URL'&$top=15&$skip='$i | jq  . | grep MediaUrl |grep -v bing |perl -pe "s/.*: \"(.*?)\".*/\1/g" |xargs -P 15 wget -T 5 -U "$UA"  -N --random-wait -P ./
done

return 0
