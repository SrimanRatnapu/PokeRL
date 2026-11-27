#!/bin/bash
# Clone the Pokemon Showdown server at the commit this project was developed against and configure it.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=5b88c476113d51a939bddbcd00b5cd182a5d8058
if [ ! -d pokemon-showdown ]; then
  git clone https://github.com/smogon/pokemon-showdown.git pokemon-showdown
fi
cd pokemon-showdown
git checkout -q "$COMMIT"
npm install
[ -f config/config.js ] || cp config/config-example.js config/config.js
# Long project paths overflow macOS's Unix-socket path limit for the REPL socket; the REPL isn't needed.
sed -i.bak 's/^exports.repl = true;/exports.repl = false;/' config/config.js && rm -f config/config.js.bak
echo "Showdown ready. Start it with: cd pokemon-showdown && node pokemon-showdown start --no-security"
