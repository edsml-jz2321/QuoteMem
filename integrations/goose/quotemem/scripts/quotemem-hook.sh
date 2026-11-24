#!/bin/sh
# QuoteMem hook for goose. Runs quotemem-hook from PATH, or the program named
# by QUOTEMEM_HOOK. Without it, the hook does nothing. stdout stays empty:
# the context goes to the file goose reads through GOOSE_MOIM_MESSAGE_FILE.
hook="${QUOTEMEM_HOOK:-quotemem-hook}"
if command -v "$hook" >/dev/null 2>&1; then
  exec "$hook" goose "$1"
fi
cat >/dev/null
exit 0
