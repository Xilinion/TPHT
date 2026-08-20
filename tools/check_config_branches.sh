#!/bin/sh
# Tripwire: the listed entry points must not test per-table configuration.
# cfg sits at the head of tpht_table_t: variant @0x0, threading @0x4,
# resize_mode @0x8.  A `cmp $imm, 0x{0,4}(%reg)` in these functions means a
# removed dispatch branch has crept back in.  chained_conc insert/put carry the
# one permitted resize_mode (@0x8) test; nothing may test variant or threading.
OBJ="$1"
fail=0
for f in flatten_tpht64_insert flatten_tpht64_get flatten_conc_tpht64_insert \
         flatten_conc_tpht64_get chained_conc_tpht64_insert chained_conc_tpht64_get \
         chained_tpht64_insert chained_tpht64_get; do
    hits=$(objdump -d "$OBJ" --disassemble="$f" 2>/dev/null | grep -cE "cmpl? +\$0x[0-9a-f]+,0x(0|4)\(%r")
    if [ "$hits" != "0" ]; then
        echo "TRIPWIRE: $f tests table configuration ($hits compare(s) vs cfg.variant/threading)"
        fail=1
    fi
done
[ "$fail" = "0" ] && echo "[tpht] config-branch tripwire clean"
exit $fail
