#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER="$ROOT/paper"
KIT="$PAPER/_kit"
BASE="https://transacl.org/tacl-submission-templates"
mkdir -p "$PAPER" "$KIT"

fetch() {
  local dest="$1" url="$2"
  if ! curl -fsSL -o "$dest" "$url"; then
    echo "In Overleaf, *New Project → Templates → search 'TACL'*; download tacl2021v1.sty and acl_natbib.bst and place them in paper/."
    exit 1
  fi
}

fetch "$PAPER/tacl2021v1.sty" "$BASE/tacl2021v1.sty"
fetch "$PAPER/acl_natbib.bst" "$BASE/acl_natbib.bst"
fetch "$KIT/tacl2021v1-template.tex" "$BASE/tacl2021v1-template.tex"
fetch "$KIT/tacl2021.bib" "$BASE/tacl2021.bib" || true

expect_sty="7c71ede112ccb3711c7073c4b1c60abfb076b83945e0e9d14ed79caa37dca3b6"
expect_bst="27a3b62fdc4211fa56fe4827572ff0524b3ba3fe96c9443faaba5a683af83c08"
expect_tex="b2c11b9c996b07d733b52f1f9c2fb2332cafab57610465d072271bea504c6021"

hash_of() { python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }

h_sty=$(hash_of "$PAPER/tacl2021v1.sty")
h_bst=$(hash_of "$PAPER/acl_natbib.bst")
h_tex=$(hash_of "$KIT/tacl2021v1-template.tex")
if [[ "$h_sty" != "$expect_sty" ]]; then
  echo "warning: tacl2021v1.sty sha256 $h_sty does not match $expect_sty"
fi
if [[ "$h_bst" != "$expect_bst" ]]; then
  echo "warning: acl_natbib.bst sha256 $h_bst does not match $expect_bst"
fi
if [[ "$h_tex" != "$expect_tex" ]]; then
  echo "warning: tacl2021v1-template.tex sha256 $h_tex does not match $expect_tex"
fi
echo "fetched TACL kit into $PAPER"
