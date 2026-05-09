"""Patch _slater_koster_pair.py for ML direct injection."""

import pathlib
import re
import sys

FILE = pathlib.Path(
    "/home/maxim/Projects/DFTB/DFTorch/src/dftorch/_slater_koster_pair.py"
)
text = FILE.read_text()
lines = text.split("\n")

# Step 1: insert _use_ml flag and _get_val_dR helper after H0/dH0 init
insert_after = None
for i, ln in enumerate(lines):
    if ln.strip().startswith("H0 = torch.zeros((HDIM * HDIM)"):
        insert_after = i + 1
        break
if insert_after is None:
    sys.exit("ERROR: could not find H0 init line")
if "dH0 = torch.zeros" in lines[insert_after]:
    insert_after += 1

helper_block = '''
    # -- ML direct-injection helpers --
    _use_ml = ml_HS_IJ is not None

    def _get_val_dR(pair_type_sel, idx_sel, dx_sel, channel, mask, direction='IJ'):
        """Return (value, dvalue_dR) for a single SK channel."""
        ch = channel + SH_shift * 10
        if _use_ml:
            src = ml_HS_IJ if direction == 'IJ' else ml_HS_JI
            dsrc = ml_HS_IJ_dR if direction == 'IJ' else ml_HS_JI_dR
            val = src[mask, ch]
            dval = dsrc[mask, ch]
        else:
            cs = coeffs_tensor[pair_type_sel, idx_sel, ch]
            val = cs[:, 0] + cs[:, 1] * dx_sel + cs[:, 2] * dx_sel**2 + cs[:, 3] * dx_sel**3
            dval = cs[:, 1] + 2 * cs[:, 2] * dx_sel + 3 * cs[:, 3] * dx_sel**2
        return val, dval
    # -- end ML helpers --'''

lines.insert(insert_after, helper_block)
text = "\n".join(lines)
lines = text.split("\n")

# Find end of first function
first_func_end = None
for i, ln in enumerate(lines):
    if "def Slater_Koster_Pair_SKF_vectorized_batch(" in ln:
        first_func_end = i
        break
if first_func_end is None:
    sys.exit("ERROR: could not find second function boundary")
print(f"First function ends at line {first_func_end}")

# Regex for coeffs_selected assignment
coeffs_re = re.compile(
    r"^(\s+)coeffs_selected\s*=\s*coeffs_tensor\[(\w+),\s*(\w+),\s*(\d+)\s*\+\s*SH_shift\s*\*\s*10\]"
)

blocks = []
i = 0
while i < first_func_end:
    ln = lines[i]
    m = coeffs_re.match(ln)
    if m:
        indent = m.group(1)
        pair_type_var = m.group(2)
        idx_var = m.group(3)
        channel = int(m.group(4))
        coeffs_line = i

        # Look ahead for polynomial evaluation
        j = i + 1
        while j < first_func_end and lines[j].strip() == "":
            j += 1

        poly_line = lines[j]
        var_m = re.match(r"^(\s+)(\w+)\s*=\s*\(?", poly_line)
        if var_m:
            var_name = var_m.group(2)
            paren_depth = poly_line.count("(") - poly_line.count(")")
            end_j = j
            while paren_depth > 0 and end_j + 1 < first_func_end:
                end_j += 1
                paren_depth += lines[end_j].count("(") - lines[end_j].count(")")

            poly_text = "\n".join(lines[j : end_j + 1])
            if "coeffs_selected" in poly_text:
                # Check for derivative block
                k = end_j + 1
                while k < first_func_end and lines[k].strip() == "":
                    k += 1

                has_deriv = False
                deriv_end = end_j
                deriv_var = None

                if k < first_func_end:
                    deriv_m = re.match(r"^(\s+)(\w+_dR)\s*=\s*\(?", lines[k])
                    if deriv_m:
                        deriv_var = deriv_m.group(2)
                        paren_depth2 = lines[k].count("(") - lines[k].count(")")
                        deriv_end2 = k
                        while paren_depth2 > 0 and deriv_end2 + 1 < first_func_end:
                            deriv_end2 += 1
                            paren_depth2 += lines[deriv_end2].count("(") - lines[
                                deriv_end2
                            ].count(")")
                        deriv_text = "\n".join(lines[k : deriv_end2 + 1])
                        if "coeffs_selected" in deriv_text:
                            has_deriv = True
                            deriv_end = deriv_end2

                # Determine dx expression
                dx_m = re.search(
                    r"coeffs_selected\[:, 1\]\s*\*\s*([\w\[\]_]+)", poly_text
                )
                dx_expr = dx_m.group(1) if dx_m else "dx"

                # Determine direction
                direction = "IJ"
                if "JI" in pair_type_var:
                    direction = "JI"
                elif pair_type_var == "sel_IJ":
                    for back in range(coeffs_line - 1, max(coeffs_line - 20, 0), -1):
                        if "sel_IJ = JI_pair_type" in lines[back]:
                            direction = "JI"
                            break
                        if "sel_IJ = IJ_pair_type" in lines[back]:
                            direction = "IJ"
                            break

                # Determine mask expression
                if pair_type_var == "IJ_pair_type" and idx_var == "idx":
                    mask_expr = "slice(None)"
                else:
                    mask_expr = "tmp_mask"

                block_info = {
                    "coeffs_line": coeffs_line,
                    "poly_start": j,
                    "poly_end": end_j,
                    "has_deriv": has_deriv,
                    "deriv_start": k if has_deriv else None,
                    "deriv_end": deriv_end,
                    "var_name": var_name,
                    "deriv_var": deriv_var if has_deriv else None,
                    "pair_type_var": pair_type_var,
                    "idx_var": idx_var,
                    "channel": channel,
                    "dx_expr": dx_expr,
                    "direction": direction,
                    "mask_expr": mask_expr,
                    "indent": indent,
                }
                blocks.append(block_info)
                print(
                    f"  Block at line {coeffs_line}: ch={channel}, var={var_name}, dir={direction}, mask={mask_expr}, has_dR={has_deriv}"
                )

                i = deriv_end + 1
                continue
    i += 1

print(f"\nFound {len(blocks)} polynomial blocks in first function")

# Replace blocks in reverse order
for blk in reversed(blocks):
    indent = blk["indent"]
    ch = blk["channel"]
    var = blk["var_name"]
    pt = blk["pair_type_var"]
    ix = blk["idx_var"]
    dx = blk["dx_expr"]
    direction = blk["direction"]
    mask = blk["mask_expr"]

    if blk["has_deriv"]:
        dvar = blk["deriv_var"]
        replacement = f"{indent}{var}, {dvar} = _get_val_dR({pt}, {ix}, {dx}, {ch}, {mask}, '{direction}')"
        start = blk["coeffs_line"]
        end = blk["deriv_end"]
    else:
        replacement = f"{indent}{var}, _ = _get_val_dR({pt}, {ix}, {dx}, {ch}, {mask}, '{direction}')"
        start = blk["coeffs_line"]
        end = blk["poly_end"]

    lines[start : end + 1] = [replacement]

text = "\n".join(lines)
FILE.write_text(text)
print(f"\nWrote patched file ({len(lines)} lines)")
