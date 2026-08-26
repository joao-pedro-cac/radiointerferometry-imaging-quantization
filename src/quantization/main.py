from ieee_754_casting import *
import json

#val = -71.17545253086246
val = -0.0

# double-precision (float64)
s, e, m  = get_double_datafields(val)
print(f"float64 value: {val:.25f}")
print(f"Sign Exp Man: {s}  {e}  {m}")
print(f"---------------------------")

# single-precision (float32)
val, error = double_to_float(val)
s, e, m  = get_float_datafields(val)
print(f"float32 value: {val:.25f}")
print(f"Sign Exp Man: {s}  {e}  {m}")
print(f"quantization error: {error * 100:.12f}%")
print(f"---------------------------")

# half-precision (float16)
val_half, error = float_to_half(val)
s, e, m  = get_half_datafields(val_half)
print(f"float16 value: {val_half:.25f}")
print(f"Sign Exp Man: {s}  {e}  {m}")
print(f"quantization error: {error * 100:.12f}%")
print(f"---------------------------")

# brain float (bfloat16)
val, error = float_to_bfloat(val)
s, e, m  = get_bfloat_datafields(val)
print(f"bfloat16 value: {val:.25f}")
print(f"Sign Exp Man: {s}  {e}  {m}")
print(f"quantization error: {error * 100:.12f}%")



DOUBLE_MIN = 5E-324
DOUBLE_MAX = 1.7976931348623157e+308

FLOAT_MIN = 1E-45
FLOAT_MAX = 3.402823466385288598117042E+38

HALF_MIN = 5.97E-8
HALF_MAX = 65504

BFLOAT_MIN = 9.1835E-41
BFLOAT_MAX = 3.3895313E+38

float_ranges = {
    "float64" : {
        "min_value" : DOUBLE_MIN,
        "max_value" : DOUBLE_MAX,
    },
    "float32" : {
        "min_value" : FLOAT_MIN,
        "max_value" : FLOAT_MAX,
    },
    "bfloat16" : {
        "min_value" : BFLOAT_MIN,
        "max_value" : BFLOAT_MAX,
    },
    "float16" : {
        "min_value" : HALF_MIN,
        "max_value" : HALF_MAX,
    }
}

with open("float-ranges.json", "w") as output_file:
    json.dump(float_ranges, output_file)
