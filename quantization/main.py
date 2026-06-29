from ieee_754_bitwise_computation import *
from numpy import float16, float32, float64
from random import random

un64 = DataUnion64Bits()
un32 = DataUnion32Bits()

value_min = -100
value_max = +100

un64.double = (value_max - value_min) * random() + value_min

sign, exponent, mantissa = get_double_datafields(un64)
print(f"float64: {un64.double:.20f}")
print(f"sign     = {sign}")
print(f"exponent = {exponent}")
print(f"mantissa = {mantissa}")
print()

un32, error = double_to_float(un64)
float_sign, float_exponent, float_mantissa = get_float_datafields(un32)

print(f"float32: {un32.float:.20f}")
print(f"sign     = {float_sign}")
print(f"exponent = {float_exponent}")
print(f"mantissa = {float_mantissa}")

print(f"\nQuantization error (from float64) = {error*100:.10f}%\n")

un32brain, error = float_to_bfloat(un32)
float_sign, float_exponent, float_mantissa = get_bfloat_datafields(un32)

print(f"bfloat16: {un32brain.float:.20f}")
print(f"sign     = {float_sign}")
print(f"exponent = {float_exponent}")
print(f"mantissa = {float_mantissa}")

print(f"\nQuantization error (from float32) = {error*100:.10f}%")

# print(f"\n\nNumPy interpretation:")
# print(f"float64({un64.double:.20f}) = {float64(un64.double):.20f}")
# print(f"float32({un64.double:.20f}) = {float32(un64.double):.20f}")
# print(f"float16({un32.float:.20f}) = {float16(un32.float):.20f}")
# print(f"float16({un64.double:.20f}) = {float16(un64.double):.20f}")