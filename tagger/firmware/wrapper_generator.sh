#!/usr/bin/env bash
#
# generate_wrapper.sh
#
# Generates an L1TSC4NGJetModel-style *_wrapper.h file from a top-level
# HLS C-synthesis header (the kind hls4ml/conifer-style tooling emits,
# with a single `void ModelName(...);` prototype).
#
# The ModelInputs / ModelOutputs structs consumed by this wrapper are
# used by other, separately-compiled .cpp files that expect them to
# ALWAYS have this exact shape:
#
#   struct ModelInputs {
#       basic_t   basic_input[N_TAGGER_PARTICLES*N_FEATURES_PARTICLES];
#       constituent_fraction_t constituent_fraction[N_TAGGER_PARTICLES];
#       jet_features_t jet_features[N_FEATURES_JETS];
#       pt_mask_t pt_mask[N_TAGGER_PARTICLES];
#   };
#
#   struct ModelOutputs {
#       nn_id_t   id_out[N_TAGGER_SCORES];
#       pt_reg_jt_t pt_out[N_PT_HEAD];
#   };
#
# So this script NEVER changes the struct field names, typedef names,
# or array sizes -- those are a fixed template, always emitted
# verbatim. Only two things vary per model header:
#
#   1. What real C++ type each canonical typedef (basic_t,
#      constituent_fraction_t, jet_features_t, pt_mask_t, nn_id_t,
#      pt_reg_jt_t) resolves to. This is taken from the model's real
#      parameter types, assigned positionally: the model's 1st input
#      parameter becomes basic_t, its 2nd becomes
#      constituent_fraction_t, and so on; the 1st output becomes
#      nn_id_t, the 2nd becomes pt_reg_jt_t.
#
#      If the model has FEWER than 4 inputs (or fewer than 2
#      outputs), the struct definition is still emitted in full (per
#      the fixed shape above) -- the leftover, unused canonical
#      typedefs just fall back to reusing the type of the last real
#      parameter, since the struct must still compile even though
#      those fields go unused.
#
#   2. Which struct fields are actually passed into the generated
#      inline `..._top()` call -- only as many as the real model
#      prototype takes, in its original parameter order. E.g. if the
#      model only takes one input array, the call only passes
#      `input.basic_input`; `constituent_fraction`, `jet_features`
#      and `pt_mask` stay declared in the struct but unused.
#
# A model with more than 4 real inputs or more than 2 real outputs
# doesn't fit this fixed template and the script will refuse to
# generate a (silently wrong) wrapper for it.
#
# FIXED-LAYOUT ADAPTATION
# The struct slots are also fixed in SIZE: the wrapper port list is always
# 16 particles x 21 basic features in, 9 id scores + 1 pt head out. A model
# variant with a SMALLER native interface (e.g. v1.0.1: 20 features, 8
# scores) is adapted inside _top(): the canonical 16x21 basic_input is
# repacked down to the model's 16x<F> layout (dropping the extra features)
# before the call, and the missing output scores are zero-filled after it.
# Array sizes are taken from the model header; non-literal factors are
# resolved through '#define name <int>' lines in the sibling defines.h.
# A model whose sizes cannot be resolved or do not fit the canonical slots
# is refused loudly.
#
# WHICH REAL PARAMETER IS AN "INPUT" VS AN "OUTPUT"?
# A prototype parameter is treated as an output if its name ends in
# `_out` (matches every example seen: layer39_out, layer40_out,
# layer22_out, ...); everything else is an input.
#
# USAGE
# -----
#   ./generate_wrapper.sh L1TSC4NGJetModel.h
#   ./generate_wrapper.sh L1TSC4NGJetModel.h -o MyWrapper.h
#   ./generate_wrapper.sh L1TSC4NGJetModel.h --io-map io_map.tsv
#
# io_map.tsv (optional) lets you override the default positional
# assignment of real parameters to canonical struct fields. Two
# tab-separated columns, no header row:
#   <header_param_name>\t<canonical_field_name>
# canonical_field_name must be one of: basic_input, constituent_fraction,
# jet_features, pt_mask (for inputs) or id_out, pt_out (for outputs).
# Any real parameter not listed falls back to the default positional
# assignment among whichever canonical slots are still free.

set -euo pipefail

# --- fixed struct template (never changes) --------------------------------
CANON_IN_FIELD=(basic_input constituent_fraction jet_features pt_mask)
CANON_IN_TYPEDEF=(basic_t constituent_fraction_t jet_features_t pt_mask_t)
CANON_IN_SIZE=("N_TAGGER_PARTICLES*N_FEATURES_PARTICLES" "N_TAGGER_PARTICLES" "N_FEATURES_JETS" "N_TAGGER_PARTICLES")

CANON_OUT_FIELD=(id_out pt_out)
CANON_OUT_TYPEDEF=(nn_id_t pt_reg_jt_t)
CANON_OUT_SIZE=("N_TAGGER_SCORES" "N_PT_HEAD")

usage() {
    grep '^#' "$0" | sed -n '2,/^set -euo/p' | sed '$d' | sed 's/^# \{0,1\}//'
    exit 1
}

HEADER=""
OUT=""
IOMAP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output) OUT="$2"; shift 2 ;;
        --io-map)    IOMAP="$2"; shift 2 ;;
        -h|--help)   usage ;;
        *)
            if [[ -z "$HEADER" ]]; then HEADER="$1"; shift
            else echo "Unexpected argument: $1" >&2; exit 1
            fi
            ;;
    esac
done

if [[ -z "$HEADER" ]]; then
    echo "Usage: $0 <input_header.h> [-o output_wrapper.h] [--io-map io_map.tsv]" >&2
    exit 1
fi
if [[ ! -f "$HEADER" ]]; then
    echo "error: no such file: $HEADER" >&2
    exit 1
fi

# --- 1. Pull out the `void Name(...);` prototype -------------------------
PROTO="$(tr '\n' ' ' < "$HEADER" | grep -oE 'void[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\([^;]*\)[[:space:]]*;' | head -n1 || true)"
if [[ -z "$PROTO" ]]; then
    echo "error: could not find a 'void Name(...);' prototype in $HEADER" >&2
    exit 1
fi

FUNC_NAME="$(echo "$PROTO" | sed -E 's/^void[[:space:]]+([A-Za-z_][A-Za-z0-9_]*).*/\1/')"
ARGS="$(echo "$PROTO" | sed -E 's/^void[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\((.*)\)[[:space:]]*;[[:space:]]*$/\1/')"

# --- 2. Split parameters on top-level commas ------------------------------
IFS=',' read -r -a RAW_PARAMS <<< "$ARGS"

PARAMS_TSV="$(mktemp)"
trap 'rm -f "$PARAMS_TSV"' EXIT

for raw in "${RAW_PARAMS[@]}"; do
    p="$(echo "$raw" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
    [[ -z "$p" ]] && continue

    if [[ "$p" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*\[[[:space:]]*(.+)[[:space:]]*\]$ ]]; then
        ptype="${BASH_REMATCH[1]}"; pname="${BASH_REMATCH[2]}"; psize="${BASH_REMATCH[3]}"
    elif [[ "$p" =~ ^([A-Za-z_][A-Za-z0-9_]*)[[:space:]]+([A-Za-z_][A-Za-z0-9_]*)$ ]]; then
        ptype="${BASH_REMATCH[1]}"; pname="${BASH_REMATCH[2]}"; psize=""
    else
        echo "error: could not parse parameter: $p" >&2
        exit 1
    fi
    printf '%s\t%s\t%s\n' "$pname" "$ptype" "$psize" >> "$PARAMS_TSV"
done

if [[ ! -s "$PARAMS_TSV" ]]; then
    echo "error: prototype has no parameters; nothing to wrap" >&2
    exit 1
fi

# --- 3. Classify into inputs / outputs, preserving prototype order -------
REAL_IN_NAME=(); REAL_IN_TYPE=(); REAL_IN_SIZE=()
REAL_OUT_NAME=(); REAL_OUT_TYPE=(); REAL_OUT_SIZE=()
while IFS=$'\t' read -r name type size; do
    if [[ "$name" =~ _out$ ]]; then
        REAL_OUT_NAME+=("$name"); REAL_OUT_TYPE+=("$type"); REAL_OUT_SIZE+=("$size")
    else
        REAL_IN_NAME+=("$name"); REAL_IN_TYPE+=("$type"); REAL_IN_SIZE+=("$size")
    fi
done < "$PARAMS_TSV"

if [[ ${#REAL_OUT_NAME[@]} -eq 0 ]]; then
    echo "error: no parameter names end in '_out'; cannot split inputs from outputs automatically" >&2
    exit 1
fi
if [[ ${#REAL_IN_NAME[@]} -eq 0 ]]; then
    echo "error: no input parameters found (all names end in '_out')" >&2
    exit 1
fi
if [[ ${#REAL_IN_NAME[@]} -gt ${#CANON_IN_FIELD[@]} ]]; then
    echo "error: model has ${#REAL_IN_NAME[@]} input parameters, but the fixed" \
         "ModelInputs template only has ${#CANON_IN_FIELD[@]} slots (${CANON_IN_FIELD[*]})." >&2
    exit 1
fi
if [[ ${#REAL_OUT_NAME[@]} -gt ${#CANON_OUT_FIELD[@]} ]]; then
    echo "error: model has ${#REAL_OUT_NAME[@]} output parameters, but the fixed" \
         "ModelOutputs template only has ${#CANON_OUT_FIELD[@]} slots (${CANON_OUT_FIELD[*]})." >&2
    exit 1
fi

# --- 4. Assign each real parameter to a canonical struct slot ------------
# IN_SLOT_TYPE[i] / OUT_SLOT_TYPE[i] = the real C++ type to typedef the
# i-th canonical field to. IN_SLOT_USED[i] tracks explicit io-map claims.
# REAL_IN_FIELD[j] / REAL_OUT_FIELD[j] = which canonical field name the
# j-th real parameter (in prototype order) was assigned to, for building
# the call line later.

n_in=${#CANON_IN_FIELD[@]}
n_out=${#CANON_OUT_FIELD[@]}
IN_SLOT_TYPE=(); IN_SLOT_USED=()
for ((i = 0; i < n_in; i++)); do IN_SLOT_TYPE[i]=""; IN_SLOT_USED[i]=0; done
OUT_SLOT_TYPE=(); OUT_SLOT_USED=()
for ((i = 0; i < n_out; i++)); do OUT_SLOT_TYPE[i]=""; OUT_SLOT_USED[i]=0; done

REAL_IN_FIELD=(); REAL_OUT_FIELD=()
for ((j = 0; j < ${#REAL_IN_NAME[@]}; j++)); do REAL_IN_FIELD[j]=""; done
for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do REAL_OUT_FIELD[j]=""; done

iomap_field_for() {
    local name="$1"
    [[ -z "$IOMAP" ]] && return 1
    awk -F'\t' -v n="$name" '$1==n{print $2; found=1} END{exit !found}' "$IOMAP"
}

# Pass 1: honor explicit io-map assignments.
for ((j = 0; j < ${#REAL_IN_NAME[@]}; j++)); do
    if field="$(iomap_field_for "${REAL_IN_NAME[j]}")"; then
        idx=-1
        for ((i = 0; i < n_in; i++)); do [[ "${CANON_IN_FIELD[i]}" == "$field" ]] && idx=$i && break; done
        if [[ $idx -eq -1 ]]; then
            echo "error: io-map field '$field' for '${REAL_IN_NAME[j]}' is not a valid input field (${CANON_IN_FIELD[*]})" >&2
            exit 1
        fi
        if [[ ${IN_SLOT_USED[idx]} -eq 1 ]]; then
            echo "error: io-map assigns more than one input parameter to '$field'" >&2
            exit 1
        fi
        IN_SLOT_TYPE[idx]="${REAL_IN_TYPE[j]}"; IN_SLOT_USED[idx]=1
        REAL_IN_FIELD[j]="$field"
    fi
done
for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
    if field="$(iomap_field_for "${REAL_OUT_NAME[j]}")"; then
        idx=-1
        for ((i = 0; i < n_out; i++)); do [[ "${CANON_OUT_FIELD[i]}" == "$field" ]] && idx=$i && break; done
        if [[ $idx -eq -1 ]]; then
            echo "error: io-map field '$field' for '${REAL_OUT_NAME[j]}' is not a valid output field (${CANON_OUT_FIELD[*]})" >&2
            exit 1
        fi
        if [[ ${OUT_SLOT_USED[idx]} -eq 1 ]]; then
            echo "error: io-map assigns more than one output parameter to '$field'" >&2
            exit 1
        fi
        OUT_SLOT_TYPE[idx]="${REAL_OUT_TYPE[j]}"; OUT_SLOT_USED[idx]=1
        REAL_OUT_FIELD[j]="$field"
    fi
done

# Pass 2: default positional assignment for whatever's left, in order.
next_free_in=0
for ((j = 0; j < ${#REAL_IN_NAME[@]}; j++)); do
    [[ -n "${REAL_IN_FIELD[j]}" ]] && continue
    while [[ $next_free_in -lt $n_in && ${IN_SLOT_USED[next_free_in]} -eq 1 ]]; do next_free_in=$((next_free_in + 1)); done
    IN_SLOT_TYPE[next_free_in]="${REAL_IN_TYPE[j]}"; IN_SLOT_USED[next_free_in]=1
    REAL_IN_FIELD[j]="${CANON_IN_FIELD[next_free_in]}"
    next_free_in=$((next_free_in + 1))
done
next_free_out=0
for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
    [[ -n "${REAL_OUT_FIELD[j]}" ]] && continue
    while [[ $next_free_out -lt $n_out && ${OUT_SLOT_USED[next_free_out]} -eq 1 ]]; do next_free_out=$((next_free_out + 1)); done
    OUT_SLOT_TYPE[next_free_out]="${REAL_OUT_TYPE[j]}"; OUT_SLOT_USED[next_free_out]=1
    REAL_OUT_FIELD[j]="${CANON_OUT_FIELD[next_free_out]}"
    next_free_out=$((next_free_out + 1))
done

# Pass 3: any canonical slots the real model doesn't use at all still need
# *some* type so the (always-fully-declared) struct compiles -- fall back
# to the last real type seen.
last_in_type="${REAL_IN_TYPE[-1]}"
for ((i = 0; i < n_in; i++)); do
    [[ -z "${IN_SLOT_TYPE[i]}" ]] && IN_SLOT_TYPE[i]="$last_in_type"
done
last_out_type="${REAL_OUT_TYPE[-1]}"
for ((i = 0; i < n_out; i++)); do
    [[ -z "${OUT_SLOT_TYPE[i]}" ]] && OUT_SLOT_TYPE[i]="$last_out_type"
done

# --- 4b. Fixed-layout adaptation ------------------------------------------
# The ModelInputs/ModelOutputs contract is FIXED (16 particles x 21 features
# basic input, 9 id scores, 1 pt head -- see the header comment). A model
# variant with a smaller interface (e.g. v1.0.1: 20 features, 8 scores) is
# adapted HERE, inside _top(): inputs are repacked from the canonical layout
# (dropping the features the model does not consume) and the missing output
# scores are zero-filled. The wrapper's PORT shape never changes.

# Numeric sizes of the canonical contract. These duplicate the N_* macros in
# data.h by design: the wrapper interface is the fixed contract this script
# is built around. If the contract changes, change both together.
CANON_N_PARTICLES=16
CANON_N_FEATURES=21
CANON_N_SCORES=9
CANON_N_PT=1
CANON_N_JET_FEATURES=2
CANON_BASIC_TOTAL=$((CANON_N_PARTICLES * CANON_N_FEATURES))

DEFINES_H="$(dirname "$HEADER")/defines.h"

# factor_value <token> -> integer (literal, or first '#define token <int>'
# in defines.h next to the model header)
factor_value() {
    local tok="$1" v
    if [[ "$tok" =~ ^[0-9]+$ ]]; then echo "$tok"; return 0; fi
    [[ -f "$DEFINES_H" ]] || return 1
    v=$(grep -m1 -E "^[[:space:]]*#define[[:space:]]+${tok}[[:space:]]+[0-9]+" "$DEFINES_H" \
        | sed -E 's/^[[:space:]]*#define[[:space:]]+[A-Za-z_][A-Za-z_0-9]*[[:space:]]+([0-9]+).*/\1/')
    [[ "$v" =~ ^[0-9]+$ ]] || return 1
    echo "$v"
}

# resolve_dim <dim-expr like '16*21' or 'N_INPUT_1_1*N_INPUT_2_1'> -> integer
resolve_dim() {
    local expr="$1" f v prod=1
    local -a FACTORS
    IFS='*' read -r -a FACTORS <<< "$expr"
    for f in "${FACTORS[@]}"; do
        f="${f//[[:space:]]/}"
        if ! v=$(factor_value "$f"); then return 1; fi
        prod=$((prod * v))
    done
    echo "$prod"
}

# Input adaptation: only the basic_input slot is repackable; every other
# input must match its canonical slot size exactly.
ADAPT_IN=0; ADAPT_IN_J=0; ADAPT_IN_TOTAL=0; ADAPT_IN_F=0
for ((j = 0; j < ${#REAL_IN_NAME[@]}; j++)); do
    field="${REAL_IN_FIELD[j]}"; sz="${REAL_IN_SIZE[j]}"
    if [[ -z "$sz" ]]; then
        echo "error: input '${REAL_IN_NAME[j]}' is not an array; cannot verify against the fixed wrapper layout" >&2
        exit 1
    fi
    if ! total=$(resolve_dim "$sz"); then
        echo "error: cannot resolve array size '$sz' of input '${REAL_IN_NAME[j]}'" \
             "(use integer literals or '#define <name> <int>' macros in defines.h)" >&2
        exit 1
    fi
    if [[ "$field" == "basic_input" ]]; then
        if (( total == CANON_BASIC_TOTAL )); then
            : # exact match: direct pass-through
        elif (( total % CANON_N_PARTICLES == 0 )) && (( total / CANON_N_PARTICLES < CANON_N_FEATURES )); then
            ADAPT_IN=1; ADAPT_IN_J=$j; ADAPT_IN_TOTAL=$total
            ADAPT_IN_F=$((total / CANON_N_PARTICLES))
        else
            echo "error: model basic input '$sz' ($total elements) cannot be derived from the canonical" \
                 "${CANON_N_PARTICLES}x${CANON_N_FEATURES} layout" >&2
            exit 1
        fi
    else
        case "$field" in
            constituent_fraction|pt_mask) canon_n=$CANON_N_PARTICLES ;;
            jet_features)                 canon_n=$CANON_N_JET_FEATURES ;;
        esac
        if (( total != canon_n )); then
            echo "error: model input '${REAL_IN_NAME[j]}' size $total != canonical $field size $canon_n" >&2
            exit 1
        fi
    fi
done

# Output adaptation: copy real outputs into canonical slots, zero-fill the
# remaining scores. A model with MORE outputs than the contract is refused.
OUT_ADAPT=(); OUT_REALN=(); OUT_CANON=()
any_out_adapt=0
for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
    field="${REAL_OUT_FIELD[j]}"; sz="${REAL_OUT_SIZE[j]}"
    if [[ "$field" == "id_out" ]]; then canon_n=$CANON_N_SCORES; else canon_n=$CANON_N_PT; fi
    if [[ -z "$sz" ]]; then
        echo "error: output '${REAL_OUT_NAME[j]}' is not an array; cannot verify against the fixed wrapper layout" >&2
        exit 1
    fi
    if ! total=$(resolve_dim "$sz"); then
        echo "error: cannot resolve array size '$sz' of output '${REAL_OUT_NAME[j]}'" \
             "(use integer literals or '#define <name> <int>' macros in defines.h)" >&2
        exit 1
    fi
    if (( total > canon_n )); then
        echo "error: model output '${REAL_OUT_NAME[j]}' has $total elements, but the fixed wrapper slot '$field' only has $canon_n" >&2
        exit 1
    fi
    OUT_REALN[j]=$total; OUT_CANON[j]=$canon_n
    if (( total < canon_n )); then OUT_ADAPT[j]=1; any_out_adapt=1; else OUT_ADAPT[j]=0; fi
done

# --- 5. Render -------------------------------------------------------------
GUARD="$(echo "$FUNC_NAME" | tr '[:lower:]' '[:upper:]')_WRAPPER_H_"
if [[ -z "$OUT" ]]; then
    DIR="$(dirname "$HEADER")"
    OUT="$DIR/${FUNC_NAME}_wrapper.h"
fi

{
    echo "#ifndef $GUARD"
    echo "#define $GUARD"
    echo
    echo "#include \"$(basename "$HEADER")\""
    echo "#include \"defines.h\""
    echo
    echo "// Defined per model, which model layers output the nnid and ptreg"
    for ((i = 0; i < n_out; i++)); do
        echo "typedef ${OUT_SLOT_TYPE[i]} ${CANON_OUT_TYPEDEF[i]};"
    done
    echo
    echo "// Defined per model, which model layers are the inputs going into"
    for ((i = 0; i < n_in; i++)); do
        echo "typedef ${IN_SLOT_TYPE[i]}   ${CANON_IN_TYPEDEF[i]};"
    done
    echo
    echo "struct ModelInputs {"
    echo "    basic_t   basic_input[N_TAGGER_PARTICLES*N_FEATURES_PARTICLES];"
    echo "    constituent_fraction_t constituent_fraction[N_TAGGER_PARTICLES];"
    echo "    jet_features_t jet_features[N_FEATURES_JETS];"
    echo "    pt_mask_t pt_mask[N_TAGGER_PARTICLES];"
    echo "};"
    echo
    echo "struct ModelOutputs {"
    echo "    nn_id_t   id_out[N_TAGGER_SCORES];"
    echo "    pt_reg_jt_t pt_out[N_PT_HEAD];"
    echo "};"
    echo
    # NOTE: do NOT emit '#pragma HLS PIPELINE II=1' here. If the model .cpp
    # uses DATAFLOW (which run_JetTagger.tcl seds to INLINE), pipelining the
    # wrapper inlines the whole net into one mega-pipeline and II explodes
    # (observed II=253). The JetTagger top-level pipeline already drives the
    # whole call chain.
    echo "inline void ${FUNC_NAME}_top( ModelInputs &input, ModelOutputs &output){"
    # NOTE: no PIPELINE pragma here. See comment above the render section:
    # pipelining the wrapper can inline the whole net into one pipeline
    # scope and break II=1.

    # Input adaptation: repack the canonical (16x21) basic_input down to the
    # model's narrower per-particle layout (features [F..20] per particle are
    # dropped; v1.0.1 consumes features 0..19).
    if (( ADAPT_IN == 1 )); then
        echo "    // Adapt fixed ${CANON_N_PARTICLES}x${CANON_N_FEATURES} basic_input to the model's ${CANON_N_PARTICLES}x${ADAPT_IN_F} layout"
        echo "    basic_t basic_in_tmp[$ADAPT_IN_TOTAL];"
        echo "    #pragma HLS ARRAY_PARTITION variable=basic_in_tmp complete dim=0"
        echo "    for (int i = 0; i < ${CANON_N_PARTICLES}; i++) {"
        echo "        #pragma HLS UNROLL"
        echo "        for (int f = 0; f < ${ADAPT_IN_F}; f++) {"
        echo "            #pragma HLS UNROLL"
        echo "            basic_in_tmp[i*${ADAPT_IN_F} + f] = input.basic_input[i*${CANON_N_FEATURES} + f];"
        echo "        }"
        echo "    }"
    fi
    # Output adaptation temps.
    for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
        if (( OUT_ADAPT[j] == 1 )); then
            slot_t="${CANON_OUT_TYPEDEF[0]}"
            [[ "${REAL_OUT_FIELD[j]}" == "pt_out" ]] && slot_t="${CANON_OUT_TYPEDEF[1]}"
            echo "    ${slot_t} ${REAL_OUT_FIELD[j]}_tmp[${OUT_REALN[j]}];"
            echo "    #pragma HLS ARRAY_PARTITION variable=${REAL_OUT_FIELD[j]}_tmp complete dim=0"
        fi
    done

    ARG_LIST=""
    for ((j = 0; j < ${#REAL_IN_NAME[@]}; j++)); do
        if (( ADAPT_IN == 1 )) && [[ $j -eq $ADAPT_IN_J ]]; then
            ARG_LIST+="basic_in_tmp, "
        else
            ARG_LIST+="input.${REAL_IN_FIELD[j]}, "
        fi
    done
    for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
        if (( OUT_ADAPT[j] == 1 )); then
            ARG_LIST+="${REAL_OUT_FIELD[j]}_tmp, "
        else
            ARG_LIST+="output.${REAL_OUT_FIELD[j]}, "
        fi
    done
    ARG_LIST="${ARG_LIST%, }"
    echo "    ${FUNC_NAME}(${ARG_LIST});"

    # Output adaptation: copy the model's outputs into the canonical slots
    # and zero-fill the scores the model does not produce (e.g. v1.0.1
    # produces 8 of the fixed 9).
    for ((j = 0; j < ${#REAL_OUT_NAME[@]}; j++)); do
        if (( OUT_ADAPT[j] == 1 )); then
            f="${REAL_OUT_FIELD[j]}"
            slot_t="${CANON_OUT_TYPEDEF[0]}"
            [[ "$f" == "pt_out" ]] && slot_t="${CANON_OUT_TYPEDEF[1]}"
            echo "    for (int k = 0; k < ${OUT_CANON[j]}; k++) {"
            echo "        #pragma HLS UNROLL"
            echo "        output.$f[k] = (k < ${OUT_REALN[j]}) ? ${f}_tmp[k] : ${slot_t}(0);"
            echo "    }"
        fi
    done
    echo "}"
    echo
    echo "#endif"
} > "$OUT"

echo "Wrote $OUT"

