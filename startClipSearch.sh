#!/bin/sh
# startClipSearch.sh
source /volume1/Code/env1/bin/activate
cd /volume1/Code/CLIP-Image-Search
gunicorn --bind 0.0.0.0:8080 web_app:server --daemon